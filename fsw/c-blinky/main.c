typedef unsigned int U32;

// Base addresses
#define RCC_BASE     0x58024400UL
#define PWR_BASE     0x58024800UL
#define GPIOE_BASE   0x58021000UL
#define FLASH_BASE   0x52002000UL
#define SYSTICK_BASE 0xE000E010UL

// RCC registers
#define RCC_CR       (*(volatile U32*) (RCC_BASE + 0x00))
#define RCC_CFGR     (*(volatile U32*) (RCC_BASE + 0x10))
#define RCC_D1CFGR   (*(volatile U32*) (RCC_BASE + 0x18))
#define RCC_D2CFGR   (*(volatile U32*) (RCC_BASE + 0x1C))
#define RCC_D3CFGR   (*(volatile U32*) (RCC_BASE + 0x20))
#define RCC_PLL1CFGR (*(volatile U32*) (RCC_BASE + 0x28))
#define RCC_PLL1DIVR (*(volatile U32*) (RCC_BASE + 0x30))
#define RCC_AHB4ENR  (*(volatile U32*) (RCC_BASE + 0xE0))

// PWR registers
#define PWR_CSR1 (*(volatile U32*) (PWR_BASE + 0x04))
#define PWR_CR3  (*(volatile U32*) (PWR_BASE + 0x0C))
#define PWR_D3CR (*(volatile U32*) (PWR_BASE + 0x18))

// GPIO registers
#define GPIOE_MODER   (*(volatile U32*) (GPIOE_BASE + 0x00))
#define GPIOE_OTYPER  (*(volatile U32*) (GPIOE_BASE + 0x04))
#define GPIOE_OSPEEDR (*(volatile U32*) (GPIOE_BASE + 0x08))
#define GPIOE_ODR     (*(volatile U32*) (GPIOE_BASE + 0x14))

// Other registers
#define FLASH_ACR   (*(volatile U32*) (FLASH_BASE + 0x00))
#define SYSTICK_CSR (*(volatile U32*) (SYSTICK_BASE + 0x00))
#define SYSTICK_RVR (*(volatile U32*) (SYSTICK_BASE + 0x04))
#define SYSTICK_CVR (*(volatile U32*) (SYSTICK_BASE + 0x08))

// Bit definitions
#define RCC_CR_HSEON             (1 << 16)
#define RCC_CR_HSERDY            (1 << 17)
#define RCC_CR_HSEBYP            (1 << 18)
#define RCC_CR_PLL1ON            (1 << 24)
#define RCC_CR_PLL1RDY           (1 << 25)
#define RCC_CFGR_SW_PLL1         (0x3 << 0)
#define RCC_CFGR_SWS_PLL1        (0x3 << 3)
#define RCC_D1CFGR_D1CPRE_DIV1   (0x0 << 8)
#define RCC_D1CFGR_HPRE_DIV2     (0x8 << 0)
#define RCC_D2CFGR_D2PPRE1_DIV2  (0x4 << 4)
#define RCC_D2CFGR_D2PPRE2_DIV2  (0x4 << 8)
#define RCC_D3CFGR_D3PPRE_DIV2   (0x4 << 4)
#define RCC_PLL1CFGR_PLL1SRC_HSE (0x2 << 0)
#define RCC_PLL1CFGR_PLL1PEN     (1 << 16)
#define RCC_PLL1CFGR_PLL1QEN     (1 << 17)
#define RCC_AHB4ENR_GPIOEEN      (1 << 4)
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

// Constants
#define RED_LED_PIN        5
#define PLL1_M             3
#define PLL1_N             100
#define PLL1_P             2
#define PLL1_Q             8
#define RTT_BUFFER_SIZE_UP 1024

// RTT structures
typedef struct {
    const char*       sName;
    char*             pBuffer;
    unsigned          SizeOfBuffer;
    unsigned          WrOff;
    volatile unsigned RdOff;
    unsigned          Flags;
} RTT_BUFFER_UP;

typedef struct {
    char          acID[16];
    int           MaxNumUpBuffers;
    int           MaxNumDownBuffers;
    RTT_BUFFER_UP aUp[1];
} RTT_CB;

// Global variables
static char   _acUpBuffer[RTT_BUFFER_SIZE_UP];
static RTT_CB _SEGGER_RTT = {
    "SEGGER RTT",
    1,
    0,
    {{"Terminal", &_acUpBuffer[0], sizeof(_acUpBuffer), 0, 0, 0}},
};
static volatile U32 systick_ms = 0;

// External symbols from linker
extern U32 _stack_end;
extern U32 _sidata, _sdata, _edata, _sbss, _ebss;

// Function declarations
int  main(void);
void Reset_Handler(void);
void Default_Handler(void);
void SysTick_Handler(void);

// Vector table
__attribute__((section(".isr_vector"))) void (*const vectors[])(void) = {
    (void*) &_stack_end,
    Reset_Handler,
    Default_Handler,
    Default_Handler,
    Default_Handler,
    Default_Handler,
    Default_Handler,
    0,
    0,
    0,
    0,
    Default_Handler,
    Default_Handler,
    0,
    Default_Handler,
    SysTick_Handler,
};

static inline void rtt_init(void) {
    _SEGGER_RTT.aUp[0] = (RTT_BUFFER_UP) {
        .sName        = "Terminal",
        .pBuffer      = _acUpBuffer,
        .SizeOfBuffer = sizeof(_acUpBuffer),
    };
}

static inline void rtt_write(const char* s) {
    if (!s) return;
    RTT_BUFFER_UP* pRing = &_SEGGER_RTT.aUp[0];
    while (*s) {
        pRing->pBuffer[pRing->WrOff] = *s++;
        pRing->WrOff                 = (pRing->WrOff + 1) % pRing->SizeOfBuffer;
    }
}

static void system_init(void) {
    PWR_CR3 &= ~PWR_CR3_BYPASS;
    PWR_CR3 &= ~PWR_CR3_SDEN;
    PWR_CR3 |= PWR_CR3_LDOEN;
    PWR_D3CR |= PWR_D3CR_VOS_SCALE1;

    U32 vos_timeout = 1000000;
    while (!(PWR_CSR1 & PWR_CSR1_ACTVOSRDY) && vos_timeout--);

    FLASH_ACR = FLASH_ACR_LATENCY_4WS | FLASH_ACR_PRFTEN | FLASH_ACR_ICEN | FLASH_ACR_DCEN;
}

static void clock_config(void) {
    RCC_CR |= RCC_CR_HSEBYP | RCC_CR_HSEON;
    U32 hse_timeout = 1000000;
    while (!(RCC_CR & RCC_CR_HSERDY) && hse_timeout--);

    RCC_D1CFGR = RCC_D1CFGR_D1CPRE_DIV1 | RCC_D1CFGR_HPRE_DIV2;
    RCC_D2CFGR = RCC_D2CFGR_D2PPRE1_DIV2 | RCC_D2CFGR_D2PPRE2_DIV2;
    RCC_D3CFGR = RCC_D3CFGR_D3PPRE_DIV2;

    RCC_PLL1CFGR = RCC_PLL1CFGR_PLL1SRC_HSE | RCC_PLL1CFGR_PLL1PEN | RCC_PLL1CFGR_PLL1QEN |
                   ((PLL1_M - 1) << 4);

    RCC_PLL1DIVR = ((PLL1_P - 1) << 9) | ((PLL1_Q - 1) << 16) | ((PLL1_N - 1) << 0);

    RCC_CR |= RCC_CR_PLL1ON;
    U32 pll_timeout = 1000000;
    while (!(RCC_CR & RCC_CR_PLL1RDY) && pll_timeout--);

    RCC_CFGR           = (RCC_CFGR & ~0x7) | RCC_CFGR_SW_PLL1;
    U32 switch_timeout = 1000000;
    while ((RCC_CFGR & 0x38) != RCC_CFGR_SWS_PLL1 && switch_timeout--);
}

static void gpio_config(void) {
    RCC_AHB4ENR |= RCC_AHB4ENR_GPIOEEN;
    // Configure PE5 (red LED) as output
    GPIOE_MODER &= ~(3 << (RED_LED_PIN * 2));
    GPIOE_MODER |= (1 << (RED_LED_PIN * 2));
    GPIOE_OTYPER &= ~(1 << RED_LED_PIN);
    GPIOE_OSPEEDR &= ~(3 << (RED_LED_PIN * 2));
}

static void systick_init(void) {
    SYSTICK_RVR = 400000 - 1;
    SYSTICK_CVR = 0;
    SYSTICK_CSR = SYSTICK_CSR_CLKSOURCE | SYSTICK_CSR_TICKINT | SYSTICK_CSR_ENABLE;
}

void SysTick_Handler(void) {
    systick_ms++;
}

static void delay_ms(U32 ms) {
    U32 start = systick_ms;
    while ((systick_ms - start) < ms);
}

void Reset_Handler(void) {
    U32* src  = &_sidata;
    U32* dest = &_sdata;

    while (dest < &_edata) *dest++ = *src++;
    dest = &_sbss;
    while (dest < &_ebss) *dest++ = 0;

    main();
    while (1);
}

void Default_Handler(void) {
    while (1);
}

int main(void) {
    rtt_init();
    system_init();
    clock_config();
    gpio_config();
    systick_init();
    while (1) {
        GPIOE_ODR |= (1 << RED_LED_PIN);
        rtt_write("LED:  ON\r\n");
        delay_ms(500);
        GPIOE_ODR &= ~(1 << RED_LED_PIN);
        rtt_write("LED: OFF\r\n");
        delay_ms(500);
    }
}
