#ifndef BOOT_H
#define BOOT_H

// boot.h — ROM (system memory) bootloader entry and reset accounting.
//
// Handover goes through a reset via a magic word in .noinit, so the ROM starts
// from a reset-state system rather than a running 64 MHz configuration.

#include "core.h"

#include "arch.h"

void boot_request_bootloader(void);
void boot_bootloader_jump_if_pending(void);

// Consecutive unexpected resets (watchdog, or a fault that reset on purpose),
// kept across resets; boot_mark_stable clears the count once the image has
// proven itself, and at BootCrashLimit the caller hands over to the ROM.
enum { BootCrashLimit = 3u };
U32 boot_crashes(void);
U32 boot_last_reset_cause(void);
void boot_mark_stable(void);
void boot_fault_reset(U32 pc, U32 lr);
U32 boot_fault_pc(void);
U32 boot_fault_lr(void);
void boot_mark_flash_programmed(void);

#ifndef BOOT__READ32
#define BOOT__READ32(addr_) (*(volatile U32 *)(addr_))
#endif

#ifndef BOOT__WRITE32
#define BOOT__WRITE32(addr_, value_) (*(volatile U32 *)(addr_) = (U32)(value_))
#endif

// Must survive Reset_Handler's .data copy and .bss clear.
#ifndef BOOT__NOINIT
#define BOOT__NOINIT __attribute__((section(".noinit"), used))
#endif

#ifndef BOOT__SYSTEM_RESET
#define BOOT__SYSTEM_RESET() arch_system_reset()
#endif

#ifndef BOOT__JUMP
#define BOOT__JUMP(stack_, entry_) do { \
    arch_systick_stop(); \
    arch_irq_disable_all(); \
    arch_vector_table_set(BOOT__SYSTEM_MEMORY_BASE); \
    arch_jump_with_stack((stack_), (entry_)); \
} while (0)
#endif

#define BOOT__SYSTEM_MEMORY_BASE 0x1FFF0000UL
#define BOOT__SRAM_BASE          0x20000000UL
#define BOOT__REQUEST_MAGIC      0xB007DF01UL
#define BOOT__FAULT_MAGIC        0xFA017A9EUL

// RM0444 §5.4.13. A software reset also drives NRST, so PINRSTF says nothing
// about an external reset; only PWRRSTF invalidates the count.
#define BOOT__RCC_CSR_ADDR   (0x40021000UL + 0x60)
#define BOOT__CSR_RMVF       (1u << 23)
#define BOOT__CSR_PINRSTF    (1u << 26)
#define BOOT__CSR_PWRRSTF    (1u << 27)
#define BOOT__CSR_SFTRSTF    (1u << 28)
#define BOOT__CSR_IWDGRSTF   (1u << 29)
#define BOOT__CSR_WWDGRSTF   (1u << 30)
#define BOOT__CSR_LPWRRSTF   (1u << 31)

// FLASH_ACR.EMPTY (PROGEMPTY), RM0444 §2.5: set at option byte loading when
// 0x0800 0000 reads erased, and while set, booting from main flash enters the
// ROM instead. Option bytes load only on a power reset or OBL_LAUNCH, so after
// a blank part's first programming every other reset would still enter the ROM.
#define BOOT__FLASH_ACR_ADDR 0x40022000UL
#define BOOT__ACR_EMPTY      (1u << 16)

// Volatile: only observed after a reset, so not a dead store.
static volatile U32 boot__request BOOT__NOINIT;
static volatile U32 boot__crashes BOOT__NOINIT;
static volatile U32 boot__crashes_check BOOT__NOINIT; // ~boot__crashes
static volatile U32 boot__fault_flag BOOT__NOINIT;
static volatile U32 boot__fault_pc BOOT__NOINIT;
static volatile U32 boot__fault_lr BOOT__NOINIT;
static U32 boot__last_cause;

void boot_request_bootloader(void) {
    boot__request = BOOT__REQUEST_MAGIC;
    arch_barrier();
    BOOT__SYSTEM_RESET();
}

U32 boot_crashes(void) {
    U32 cause = BOOT__READ32(BOOT__RCC_CSR_ADDR);
    BOOT__WRITE32(BOOT__RCC_CSR_ADDR, cause | BOOT__CSR_RMVF);
    boot__last_cause = cause;

    bool fault = boot__fault_flag == BOOT__FAULT_MAGIC;
    boot__fault_flag = 0;
    bool valid = boot__crashes_check == ~boot__crashes;
    bool unexpected = fault || (cause & (BOOT__CSR_IWDGRSTF | BOOT__CSR_WWDGRSTF | BOOT__CSR_LPWRRSTF)) != 0u;

    if (!valid || (cause & BOOT__CSR_PWRRSTF) != 0u) boot__crashes = 0;
    else if (unexpected) boot__crashes = boot__crashes + 1u;
    boot__crashes_check = ~boot__crashes;
    return boot__crashes;
}

U32 boot_last_reset_cause(void) {
    return boot__last_cause;
}

void boot_mark_stable(void) {
    boot__crashes = 0;
    boot__crashes_check = ~0u;
}

void boot_fault_reset(U32 pc, U32 lr) {
    boot__fault_pc = pc;
    boot__fault_lr = lr;
    boot__fault_flag = BOOT__FAULT_MAGIC;
    arch_barrier();
    BOOT__SYSTEM_RESET();
}

U32 boot_fault_pc(void) {
    return boot__fault_pc;
}

U32 boot_fault_lr(void) {
    return boot__fault_lr;
}

// Running, so flash holds an image: clear EMPTY as the next option byte load
// would, so later resets come back here.
void boot_mark_flash_programmed(void) {
    U32 acr = BOOT__READ32(BOOT__FLASH_ACR_ADDR);
    if ((acr & BOOT__ACR_EMPTY) != 0u) BOOT__WRITE32(BOOT__FLASH_ACR_ADDR, acr & ~BOOT__ACR_EMPTY);
}

void boot_bootloader_jump_if_pending(void) {
    if (boot__request != BOOT__REQUEST_MAGIC) return;

    // Consume first: the next reset must land in the application.
    boot__request = 0;
    arch_barrier();

    U32 stack = BOOT__READ32(BOOT__SYSTEM_MEMORY_BASE);
    U32 entry = BOOT__READ32(BOOT__SYSTEM_MEMORY_BASE + 4u);

    // Refuse an implausible vector rather than branch into nothing.
    if ((stack & 0xFFFF0000UL) != BOOT__SRAM_BASE) return;
    if ((entry & 0xFFFF0000UL) != BOOT__SYSTEM_MEMORY_BASE) return;

    BOOT__JUMP(stack, entry);
}

#endif
