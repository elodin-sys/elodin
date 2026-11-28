#ifndef ARCH_H
#define ARCH_H

#include "core.h"

#define ARCH__REG32(addr) (*(volatile U32 *)(addr))

#define ARCH__SYSTICK_BASE 0xE000E010UL
#define ARCH__SYSTICK_CSR  ARCH__REG32(ARCH__SYSTICK_BASE + 0x00)
#define ARCH__SYSTICK_RVR  ARCH__REG32(ARCH__SYSTICK_BASE + 0x04)
#define ARCH__SYSTICK_CVR  ARCH__REG32(ARCH__SYSTICK_BASE + 0x08)

#define ARCH__NVIC_ISER    ARCH__REG32(0xE000E100UL)
#define ARCH__NVIC_ICER    ARCH__REG32(0xE000E180UL)
#define ARCH__NVIC_ICPR    ARCH__REG32(0xE000E280UL)
#define ARCH__NVIC_IPR(irq_) (*(volatile U8 *)(0xE000E400UL + (irq_)))

#define ARCH__SCB_ICSR     ARCH__REG32(0xE000ED04UL)
#define ARCH__SCB_VTOR     ARCH__REG32(0xE000ED08UL)
#define ARCH__ICSR_PENDSTCLR (1u << 25)
#define ARCH__ICSR_PENDSVCLR (1u << 27)
#define ARCH__SCB_AIRCR    ARCH__REG32(0xE000ED0CUL)
#define ARCH__SCB_DHCSR    ARCH__REG32(0xE000EDF0UL)

#define ARCH__SYSTICK_CSR_ENABLE    (1u << 0)
#define ARCH__SYSTICK_CSR_TICKINT   (1u << 1)
#define ARCH__SYSTICK_CSR_CLKSOURCE (1u << 2)
#define ARCH__SYSTICK_CSR_COUNTFLAG (1u << 16)

#define ARCH__SCB_AIRCR_VECTKEY     (0x5FAu << 16)
#define ARCH__SCB_AIRCR_SYSRESETREQ (1u << 2)

#define ARCH__SCB_DHCSR_C_DEBUGEN   (1u << 0)

#define ARCH__NVIC_PRIO_BITS 2u

typedef U8 Arch_Irq;

#define ARCH_IRQ_NONE 0xFFu

static void arch_barrier(void) {
    __asm volatile ("" ::: "memory");
}

static U32 arch_irq_save(void) {
    U32 primask;
    __asm volatile (
        "mrs %0, primask\n"
        "cpsid i\n"
        : "=r"(primask)
        :
        : "memory"
    );
    return primask;
}

static void arch_irq_restore(U32 primask) {
    __asm volatile (
        "msr primask, %0\n"
        :
        : "r"(primask)
        : "memory"
    );
}

static U32 arch_exception_number(void) {
    U32 ipsr;
    __asm volatile ("mrs %0, ipsr" : "=r"(ipsr));
    return ipsr & 0x1FFu;
}

static inline U32 arch_main_stack_pointer(void) {
    U32 msp;
    __asm volatile ("mrs %0, msp" : "=r"(msp));
    return msp;
}

static inline U32 arch_process_stack_pointer(void) {
    U32 psp;
    __asm volatile ("mrs %0, psp" : "=r"(psp));
    return psp;
}

static bool arch_debugger_attached(void) {
    return (ARCH__SCB_DHCSR & ARCH__SCB_DHCSR_C_DEBUGEN) != 0;
}

static void arch_breakpoint(void) {
    __asm volatile ("bkpt #0" ::: "memory");
}

static inline bool arch_irq_valid(Arch_Irq irq) {
    return irq != ARCH_IRQ_NONE && irq <= 30u;
}

static inline void arch_irq_enable(Arch_Irq irq) {
    if (!arch_irq_valid(irq)) return;
    ARCH__NVIC_ISER = 1u << (U32)irq;
}

static inline void arch_irq_disable(Arch_Irq irq) {
    if (!arch_irq_valid(irq)) return;
    ARCH__NVIC_ICER = 1u << (U32)irq;
}

static inline void arch_irq_set_priority(Arch_Irq irq, U8 priority) {
    if (!arch_irq_valid(irq)) return;

    U8 mask = (1u << ARCH__NVIC_PRIO_BITS) - 1u;
    ARCH__NVIC_IPR((U32)irq) = (priority & mask) << (8u - ARCH__NVIC_PRIO_BITS);
}

static inline void arch_irq_disable_all(void) {
    ARCH__NVIC_ICER = 0xFFFFFFFFu;
    ARCH__NVIC_ICPR = 0xFFFFFFFFu;
    arch_barrier();
}

static inline void arch_systick_stop(void) {
    ARCH__SYSTICK_CSR = 0;
    ARCH__SYSTICK_RVR = 0;
    ARCH__SYSTICK_CVR = 0;
    arch_barrier();
}

static inline void arch_vector_table_set(U32 base) {
    ARCH__SCB_VTOR = base;
    arch_barrier();
}

// First thing in Reset_Handler. Leaving DFU, the ROM jumps here without a reset
// (AN3156 §5.5), with address 0 still aliasing the ROM and its SysTick, NVIC
// and PRIMASK state in place; restore the reset state, VTOR on our own table.
static inline void arch_startup_state(U32 vector_table) {
    __asm volatile ("cpsid i" ::: "memory");
    arch_systick_stop();
    arch_irq_disable_all();
    ARCH__SCB_ICSR = ARCH__ICSR_PENDSTCLR | ARCH__ICSR_PENDSVCLR;
    arch_vector_table_set(vector_table);
    __asm volatile ("cpsie i" ::: "memory");
}

// Reloads MSP and branches to `entry`, keeping its Thumb bit. Never returns.
static inline void arch_jump_with_stack(U32 stack, U32 entry) {
    __asm volatile (
        "msr msp, %0\n"
        "bx %1\n"
        :
        : "r"(stack), "r"(entry)
        : "memory"
    );

    while (1) {
    }
}

static void arch_system_reset(void) {
    arch_barrier();
    ARCH__SCB_AIRCR = ARCH__SCB_AIRCR_VECTKEY | ARCH__SCB_AIRCR_SYSRESETREQ;
    arch_barrier();

    while (1) {
    }
}

static void arch_wait_for_interrupt(void) {
    __asm volatile ("wfi" ::: "memory");
}

#endif
