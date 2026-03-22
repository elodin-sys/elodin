typedef unsigned int U32;
typedef unsigned long long U64;
typedef unsigned char U8;

// Base addresses
#define RCC_BASE     0x58024400UL
#define PWR_BASE     0x58024800UL
#define GPIOA_BASE   0x58020000UL
#define GPIOE_BASE   0x58021000UL
#define FLASH_BASE   0x52002000UL
#define SYSTICK_BASE 0xE000E010UL
#define USART1_BASE  0x40011000UL

// RCC registers
#define RCC_CR        (*(volatile U32*) (RCC_BASE + 0x00))
#define RCC_CFGR      (*(volatile U32*) (RCC_BASE + 0x10))
#define RCC_D1CFGR    (*(volatile U32*) (RCC_BASE + 0x18))
#define RCC_D2CFGR    (*(volatile U32*) (RCC_BASE + 0x1C))
#define RCC_D3CFGR    (*(volatile U32*) (RCC_BASE + 0x20))
#define RCC_AHB4ENR   (*(volatile U32*) (RCC_BASE + 0xE0))
#define RCC_APB2ENR   (*(volatile U32*) (RCC_BASE + 0xF0))

// PWR registers
#define PWR_CSR1 (*(volatile U32*) (PWR_BASE + 0x04))
#define PWR_CR3  (*(volatile U32*) (PWR_BASE + 0x0C))
#define PWR_D3CR (*(volatile U32*) (PWR_BASE + 0x18))

// GPIOA registers
#define GPIOA_MODER (*(volatile U32*) (GPIOA_BASE + 0x00))
#define GPIOA_AFRH  (*(volatile U32*) (GPIOA_BASE + 0x24))

// GPIOE registers
#define GPIOE_MODER   (*(volatile U32*) (GPIOE_BASE + 0x00))
#define GPIOE_OTYPER  (*(volatile U32*) (GPIOE_BASE + 0x04))
#define GPIOE_OSPEEDR (*(volatile U32*) (GPIOE_BASE + 0x08))
#define GPIOE_ODR     (*(volatile U32*) (GPIOE_BASE + 0x14))

// USART1 registers
#define USART1_CR1 (*(volatile U32*) (USART1_BASE + 0x00))
#define USART1_BRR (*(volatile U32*) (USART1_BASE + 0x0C))
#define USART1_ISR (*(volatile U32*) (USART1_BASE + 0x1C))
#define USART1_TDR (*(volatile U32*) (USART1_BASE + 0x28))

// Other registers
#define FLASH_ACR   (*(volatile U32*) (FLASH_BASE + 0x00))
#define SYSTICK_CSR (*(volatile U32*) (SYSTICK_BASE + 0x00))
#define SYSTICK_RVR (*(volatile U32*) (SYSTICK_BASE + 0x04))
#define SYSTICK_CVR (*(volatile U32*) (SYSTICK_BASE + 0x08))

// Bit definitions
#define RCC_CR_PLL1ON            (1 << 24)
#define RCC_CFGR_SW_PLL1         (0x3 << 0)
#define RCC_AHB4ENR_GPIOAEN      (1 << 0)
#define RCC_AHB4ENR_GPIOEEN      (1 << 4)
#define RCC_APB2ENR_USART1EN     (1 << 4)
#define PWR_CR3_LDOEN            (1 << 1)
#define PWR_CR3_BYPASS           (1 << 0)
#define PWR_CR3_SDEN             (1 << 2)
#define PWR_D3CR_VOS_SCALE1      (0x3 << 14)
#define PWR_CSR1_ACTVOSRDY       (1 << 13)
#define FLASH_ACR_LATENCY_4WS    (4 << 0)
#define FLASH_ACR_PRFTEN         (1 << 8)
#define FLASH_ACR_ICEN           (1 << 9)
#define FLASH_ACR_DCEN           (1 << 10)
#define SYSTICK_CSR_ENABLE       (1 << 0)
#define SYSTICK_CSR_TICKINT      (1 << 1)
#define SYSTICK_CSR_CLKSOURCE    (1 << 2)
#define USART1_ISR_TXE           (1 << 7)
#define USART1_CR1_UE            (1 << 0)
#define USART1_CR1_TE            (1 << 3)

#define RED_LED_PIN 3

#define WAIT_FOR(condition, error_msg) do { \
    U32 timeout = 1000000; \
    while (!(condition) && timeout--); \
    if (!timeout) { panic(error_msg); } \
} while(0)

// Run directly from HSI at 64MHz (no PLL needed for LED + UART)
static const U32 SYSTICK_FREQ = 64000000;
static const U32 APB2_FREQ    = 64000000;

// External symbols from linker
extern U32 _stack_end;
extern U32 _sidata, _sdata, _edata, _sbss, _ebss;

// Function declarations
int  main(void);
void Reset_Handler(void);
void Default_Handler(void);
void Fault_Handler(void);

__attribute__((section(".isr_vector"))) void (*const vectors[])(void) = {
    (void*) &_stack_end,
    Reset_Handler,
    Fault_Handler,        // NMI
    Fault_Handler,        // Hard Fault
    Fault_Handler,        // MemManage
    Fault_Handler,        // Bus Fault
    Fault_Handler,        // Usage Fault
    0, 0, 0, 0,
    Default_Handler,      // SVCall
    Default_Handler,      // Debug Monitor
    0,
    Default_Handler,      // PendSV
    0,                    // SysTick
};

static void panic(const char* msg) {
    (void)msg;
    GPIOE_ODR |= (1 << RED_LED_PIN);
    __asm("bkpt #0");
    while (1);
}

static void system_init(void) {
    PWR_CR3 &= ~PWR_CR3_BYPASS;
    PWR_CR3 &= ~PWR_CR3_SDEN;
    PWR_CR3 |= PWR_CR3_LDOEN;
    PWR_D3CR |= PWR_D3CR_VOS_SCALE1;
    WAIT_FOR(PWR_CSR1 & PWR_CSR1_ACTVOSRDY, "VOS");
    FLASH_ACR = FLASH_ACR_LATENCY_4WS | FLASH_ACR_PRFTEN | FLASH_ACR_ICEN | FLASH_ACR_DCEN;
}

static void clock_config(void) {
    // Stay on HSI at 64MHz. Disable PLL1 if the bootloader left it running.
    RCC_CFGR &= ~RCC_CFGR_SW_PLL1;
    RCC_CR &= ~RCC_CR_PLL1ON;
    // All bus prescalers at /1 (default after reset)
    RCC_D1CFGR = 0;
    RCC_D2CFGR = 0;
    RCC_D3CFGR = 0;
    // 1 wait state is sufficient for 64MHz at VOS1
    FLASH_ACR = (1 << 0) | FLASH_ACR_PRFTEN | FLASH_ACR_ICEN | FLASH_ACR_DCEN;
}

static void gpio_config(void) {
    RCC_AHB4ENR |= RCC_AHB4ENR_GPIOEEN;
    GPIOE_MODER &= ~(3 << (RED_LED_PIN * 2));
    GPIOE_MODER |= (1 << (RED_LED_PIN * 2));
    GPIOE_OTYPER &= ~(1 << RED_LED_PIN);
    GPIOE_OSPEEDR &= ~(3 << (RED_LED_PIN * 2));
}

// --- USART1 on PA9 (TX) at 115200 8N1 ---

static void uart_init(void) {
    RCC_AHB4ENR |= RCC_AHB4ENR_GPIOAEN;
    RCC_APB2ENR |= RCC_APB2ENR_USART1EN;

    // PA9 = AF7 (USART1_TX): MODER bits [19:18] = 10 (AF)
    GPIOA_MODER &= ~(3U << 18);
    GPIOA_MODER |=  (2U << 18);
    // AFRH bits [7:4] = 7 (AF7 for PA9)
    GPIOA_AFRH &= ~(0xFU << 4);
    GPIOA_AFRH |=  (7U << 4);

    USART1_BRR = APB2_FREQ / 115200;  // 868
    USART1_CR1 = USART1_CR1_TE | USART1_CR1_UE;
}

static void uart_putchar(U8 c) {
    while (!(USART1_ISR & USART1_ISR_TXE));
    USART1_TDR = c;
}

static void uart_write(const U8* data, int len) {
    for (int i = 0; i < len; i++)
        uart_putchar(data[i]);
}

// --- COBS encoding ---

static int cobs_encode(const U8* input, int len, U8* output) {
    int read_idx  = 0;
    int write_idx = 1;
    int code_idx  = 0;
    U8  code      = 1;

    while (read_idx < len) {
        if (input[read_idx] == 0) {
            output[code_idx] = code;
            code     = 1;
            code_idx = write_idx++;
            read_idx++;
        } else {
            output[write_idx++] = input[read_idx++];
            code++;
            if (code == 0xFF) {
                output[code_idx] = code;
                code     = 1;
                code_idx = write_idx++;
            }
        }
    }
    output[code_idx] = code;
    return write_idx;
}

// --- EL log frame: 0x00 | COBS(['E','L',ver,kind,level,msg...]) | 0x00 ---

#define EL_LOG_LEVEL_INFO 2

static void uart_log(U8 level, const char* msg) {
    U8 frame[128];
    U8 encoded[132];

    int len = 0;
    frame[len++] = 'E';
    frame[len++] = 'L';
    frame[len++] = 1;   // version
    frame[len++] = 1;   // kind = LOG
    frame[len++] = level;

    while (*msg && len < (int)sizeof(frame))
        frame[len++] = (U8)*msg++;

    int enc_len = cobs_encode(frame, len, encoded);

    uart_putchar(0x00);
    uart_write(encoded, enc_len);
    uart_putchar(0x00);
}

// --- Delay ---

static void delay_us(U32 us) {
    U32 ticks = ((U64)us * SYSTICK_FREQ) / 1000000;
    if (ticks > 1) {
        SYSTICK_RVR = ticks - 1;
        SYSTICK_CVR = 0;
        SYSTICK_CSR = SYSTICK_CSR_CLKSOURCE | SYSTICK_CSR_ENABLE;
        while (!(SYSTICK_CSR & (1 << 16)));
        SYSTICK_CSR = 0;
    }
}

static void delay_ms(U32 ms) {
    while (ms > 4294) {
        delay_us(4294000);
        ms -= 4294;
    }
    delay_us(ms * 1000);
}

// --- Entry points ---

void Reset_Handler(void) {
    U32* src  = &_sidata;
    U32* dest = &_sdata;
    while (dest < &_edata) *dest++ = *src++;
    dest = &_sbss;
    while (dest < &_ebss) *dest++ = 0;
    main();
    while (1);
}

void Default_Handler(void) { panic("IRQ"); }
void Fault_Handler(void)   { panic("FAULT"); }

int main(void) {
    system_init();
    clock_config();
    gpio_config();
    uart_init();

    while (1) {
        GPIOE_ODR |= (1 << RED_LED_PIN);
        uart_log(EL_LOG_LEVEL_INFO, "LED:  ON");
        delay_ms(500);
        GPIOE_ODR &= ~(1 << RED_LED_PIN);
        uart_log(EL_LOG_LEVEL_INFO, "LED: OFF");
        delay_ms(500);
    }
}
