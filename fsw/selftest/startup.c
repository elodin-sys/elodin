// startup.c — reset, vector table and fault handler for the M7 core.

#include "board.h"
#include "hw.h"

extern uint32_t _estack, _sidata, _sdata, _edata, _sbss, _ebss;
extern const uint32_t vectors[];
int main(void);

void Reset_Handler(void) {
    // VDDSMPS is grounded: leave Run* mode on the LDO before anything else
    // (ST ExitRun0Mode, USE_PWR_LDO_SUPPLY with SMPS present). PWR_CR3 is
    // write-once per power cycle; selecting SMPS here would lock the part up.
    PWR->CR3 = (PWR->CR3 & ~PWR_CR3_SMPSEN) | PWR_CR3_LDOEN;
    while ((PWR->CSR1 & PWR_CSR1_ACTVOSRDY) == 0u) {
    }

    SCB->VTOR = (uint32_t)vectors;
    for (uint32_t *s = &_sidata, *d = &_sdata; d < &_edata;) *d++ = *s++;
    for (uint32_t *d = &_sbss; d < &_ebss;) *d++ = 0u;
    __enable_irq();
    main();
    while (1) {
    }
}

// Any fault: red LED on, a note on the console if it is up, then stop.
void Fault_Handler(void) {
    __disable_irq();
    RCC->AHB4ENR |= RCC_AHB4ENR_GPIOEEN;
    pin_output(LED_RED, true, false);
    if ((RCC->APB1LENR & RCC_APB1LENR_USART3EN) != 0u) {
        con_printf("\n[FAULT] CFSR 0x%08x HFSR 0x%08x\n", SCB->CFSR, SCB->HFSR);
    }
    while (1) {
    }
}

__attribute__((section(".isr_vector"), used)) const uint32_t vectors[16 + 150] = {
    [0] = (uint32_t)&_estack,
    [1] = (uint32_t)Reset_Handler,
    [2] = (uint32_t)Fault_Handler,   // NMI
    [3] = (uint32_t)Fault_Handler,   // HardFault
    [4] = (uint32_t)Fault_Handler,   // MemManage
    [5] = (uint32_t)Fault_Handler,   // BusFault
    [6] = (uint32_t)Fault_Handler,   // UsageFault
    [11] = (uint32_t)Fault_Handler,  // SVCall
    [14] = (uint32_t)Fault_Handler,  // PendSV
    [15] = (uint32_t)SysTick_Handler,
    // No peripheral interrupts are enabled.
};

// GCC may emit calls to these for struct initialisation and copies.
void *memset(void *d, int c, unsigned int n) {
    unsigned char *p = d;
    while (n--) *p++ = (unsigned char)c;
    return d;
}
