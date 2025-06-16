typedef unsigned int U32;
typedef unsigned long long U64;

// Base addresses
#define RCC_BASE     0x58024400UL
#define PWR_BASE     0x58024800UL
#define GPIOE_BASE   0x58021000UL
#define FLASH_BASE   0x52002000UL
#define SYSTICK_BASE 0xE000E010UL

// RCC registers
#define RCC_CR        (*(volatile U32*) (RCC_BASE + 0x00))
#define RCC_CFGR      (*(volatile U32*) (RCC_BASE + 0x10))
#define RCC_D1CFGR    (*(volatile U32*) (RCC_BASE + 0x18))
#define RCC_D2CFGR    (*(volatile U32*) (RCC_BASE + 0x1C))
#define RCC_D3CFGR    (*(volatile U32*) (RCC_BASE + 0x20))
#define RCC_PLLCKSELR (*(volatile U32*) (RCC_BASE + 0x24))
#define RCC_PLL1CFGR  (*(volatile U32*) (RCC_BASE + 0x28))
#define RCC_PLL1DIVR  (*(volatile U32*) (RCC_BASE + 0x30))
#define RCC_AHB4ENR   (*(volatile U32*) (RCC_BASE + 0xE0))

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
#define RCC_D1CFGR_D1PPRE_DIV2   (0x4 << 4)
#define RCC_D2CFGR_D2PPRE1_DIV2  (0x4 << 4)
#define RCC_D2CFGR_D2PPRE2_DIV2  (0x4 << 8)
#define RCC_D3CFGR_D3PPRE_DIV2   (0x4 << 4)
#define RCC_PLLCKSELR_PLLSRC_HSE (0x2 << 0)
#define RCC_PLL1CFGR_PLL1PEN     (1 << 16)
#define RCC_PLL1CFGR_PLL1QEN     (1 << 17)
#define RCC_PLL1CFGR_PLL1RGE     (0x2 << 2)   // Input frequency range: 8-16MHz
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

// Timeout helper macro
#define WAIT_FOR(condition, error_msg) do { \
    U32 timeout = 1000000; \
    while (!(condition) && timeout--); \
    if (!timeout) { panic(error_msg); } \
} while(0)

// PLL1 configuration: 24MHz HSE ÷ M × N ÷ P = SYSCLK
// 24MHz ÷ 3 × 100 ÷ 2 = 400MHz SYSCLK
// VCO = 24MHz ÷ 3 × 100 = 800MHz
#define PLL1_M             3    // Input divider
#define PLL1_N             100  // VCO multiplier
#define PLL1_P             2    // SYSCLK divider (400MHz)
#define PLL1_Q             8    // Other outputs divider (100MHz)
#define RTT_BUFFER_SIZE_UP 1024

static const U32 SYSTICK_FREQ = 400000000;  // 400MHz d1cpreclk

/*
 * RTT (Real-Time Transfer) structures based on SEGGER RTT protocol
 * Copyright (c) SEGGER Microcontroller GmbH
 * Used under BSD-style license - see https://github.com/SEGGERMicro/RTT/blob/master/RTT/SEGGER_RTT.h
 */

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
void Fault_Handler(void);
void SysTick_Handler(void);

// Vector table - ARM Cortex-M7 exception handlers
__attribute__((section(".isr_vector"))) void (*const vectors[])(void) = {
    (void*) &_stack_end,  // 0: Initial stack pointer
    Reset_Handler,        // 1: Reset
    Fault_Handler,        // 2: NMI
    Fault_Handler,        // 3: Hard Fault
    Fault_Handler,        // 4: Memory Management Fault
    Fault_Handler,        // 5: Bus Fault
    Fault_Handler,        // 6: Usage Fault
    0,                    // 7: Reserved
    0,                    // 8: Reserved
    0,                    // 9: Reserved
    0,                    // 10: Reserved
    Default_Handler,      // 11: SVCall
    Default_Handler,      // 12: Debug Monitor
    0,                    // 13: Reserved
    Default_Handler,      // 14: PendSV
    0,                    // 15: SysTick Timer (not used - polling mode)
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

static void panic(const char* msg) {
    // Turn on red LED to indicate fault
    GPIOE_ODR |= (1 << RED_LED_PIN);
    rtt_write(msg);
    rtt_write("\r\n");
    __asm("bkpt #0");
    while (1);
}

static void system_init(void) {
    PWR_CR3 &= ~PWR_CR3_BYPASS;       // Use internal regulator (not external supply)
    PWR_CR3 &= ~PWR_CR3_SDEN;         // For internal regulator, use LDO (not SMPS)
    PWR_CR3 |= PWR_CR3_LDOEN;         // Enable the internal LDO
    PWR_D3CR |= PWR_D3CR_VOS_SCALE1;  // Set voltage scale to 1

    WAIT_FOR(PWR_CSR1 & PWR_CSR1_ACTVOSRDY, "FAULT: VOS failed to stabilize");

    FLASH_ACR = FLASH_ACR_LATENCY_4WS | FLASH_ACR_PRFTEN | FLASH_ACR_ICEN | FLASH_ACR_DCEN;  // 4 wait states, enable caches
}

static void clock_config(void) {
    // Enable 24MHz HSE (external clock, bypass mode)
    RCC_CR |= RCC_CR_HSEBYP | RCC_CR_HSEON;
    WAIT_FOR(RCC_CR & RCC_CR_HSERDY, "FAULT: HSE failed to start");

    // Disable PLL1 before configuration
    RCC_CR &= ~RCC_CR_PLL1ON;
    WAIT_FOR(!(RCC_CR & RCC_CR_PLL1RDY), "FAULT: PLL1 failed to turn off");
    // Configure PLL source and input divider: HSE + DIVM
    RCC_PLLCKSELR = RCC_PLLCKSELR_PLLSRC_HSE | (PLL1_M << 4);
    // Configure PLL1: enable outputs + set input frequency range
    // PLL1RGE: 8-16MHz input range (our input is 8MHz)
    // PLL1VCOSEL: Wide VCO range (default bit=0 for VCO=800MHz)
    RCC_PLL1CFGR = RCC_PLL1CFGR_PLL1PEN | RCC_PLL1CFGR_PLL1QEN | RCC_PLL1CFGR_PLL1RGE;
    // Configure PLL1 dividers: 24MHz ÷ 3 × 100 ÷ 2 = 400MHz
    RCC_PLL1DIVR = ((PLL1_P - 1) << 9) | ((PLL1_Q - 1) << 16) | ((PLL1_N - 1) << 0);
    // Enable PLL1 and wait for lock
    RCC_CR |= RCC_CR_PLL1ON;
    WAIT_FOR(RCC_CR & RCC_CR_PLL1RDY, "FAULT: PLL1 failed to lock");

    // Switch SYSCLK to PLL1-P
    RCC_CFGR |= RCC_CFGR_SW_PLL1;
    // SYSCLK=400MHz, HCLK=SYSCLK/2=200MHz, APB1/2/3=HCLK/2=100MHz
    RCC_D1CFGR = RCC_D1CFGR_D1CPRE_DIV1 | RCC_D1CFGR_HPRE_DIV2 | RCC_D1CFGR_D1PPRE_DIV2;
    RCC_D2CFGR = RCC_D2CFGR_D2PPRE1_DIV2 | RCC_D2CFGR_D2PPRE2_DIV2;
    RCC_D3CFGR = RCC_D3CFGR_D3PPRE_DIV2;
}

static void gpio_config(void) {
    RCC_AHB4ENR |= RCC_AHB4ENR_GPIOEEN;
    // Configure PE5 (red LED) as output
    GPIOE_MODER &= ~(3 << (RED_LED_PIN * 2));
    GPIOE_MODER |= (1 << (RED_LED_PIN * 2));
    GPIOE_OTYPER &= ~(1 << RED_LED_PIN);
    GPIOE_OSPEEDR &= ~(3 << (RED_LED_PIN * 2));
}

static void delay_us(U32 us) {
    U32 ticks = ((U64)us * SYSTICK_FREQ) / 1000000;
    if (ticks > 1) {
        SYSTICK_RVR = ticks - 1;
        SYSTICK_CVR = 0;
        SYSTICK_CSR = SYSTICK_CSR_CLKSOURCE | SYSTICK_CSR_ENABLE;
        while (!(SYSTICK_CSR & (1 << 16)));  // Poll until counter wraps (COUNTFLAG set)
        SYSTICK_CSR = 0;  // Disable counter
    }
}

static void delay_ms(U32 ms) {
    // Handle large delays by breaking into smaller chunks
    while (ms > 4294) {
        delay_us(4294000);  // ~4.3 seconds max per chunk
        ms -= 4294;
    }
    delay_us(ms * 1000);
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
    panic("ERROR: Unhandled interrupt");
}

void Fault_Handler(void) {
    panic("FAULT: System exception occurred");
}

int main(void) {
    rtt_init();
    system_init();
    clock_config();
    gpio_config();
    while (1) {
        GPIOE_ODR |= (1 << RED_LED_PIN);
        rtt_write("LED:  ON\r\n");
        delay_ms(500);
        GPIOE_ODR &= ~(1 << RED_LED_PIN);
        rtt_write("LED: OFF\r\n");
        delay_ms(500);
    }
}
