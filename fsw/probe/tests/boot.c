// boot.c — host tests for bootloader handover and reset accounting, against a
// modeled reset-cause register and system memory vector.

#include "test.h"

#define ARCH_H
static void arch_barrier(void);

enum {
    FlashAcr = 0x40022000u,
    RccCsr = 0x40021060u,
    SystemMemory = 0x1FFF0000u,
};

typedef struct {
    U32 system_memory[2];
    U32 csr;
    U32 acr;
    U32 acr_writes;
    U32 resets;
    U32 jumps;
    U32 jump_stack;
    U32 jump_entry;
} Model;

static Model model;

static void model_reset(void) {
    model = (Model){ .system_memory = { 0x20001000u, 0x1FFF3421u } };
}

static U32 model_read(UPtr address) {
    switch ((U32)address) {
    case RccCsr:            return model.csr;
    case FlashAcr:          return model.acr;
    case SystemMemory:      return model.system_memory[0];
    case SystemMemory + 4u: return model.system_memory[1];
    default:
        CHECK(false);
        return 0;
    }
}

static void model_write(UPtr address, U32 value) {
    if ((U32)address == FlashAcr) {
        model.acr = value;
        model.acr_writes++;
        return;
    }
    CHECK((U32)address == RccCsr);
    if (value & (1u << 23)) model.csr &= 0x007FFFFFu; // RMVF
}

static void model_system_reset(void) {
    model.resets++;
}

static void model_jump(U32 stack, U32 entry) {
    model.jumps++;
    model.jump_stack = stack;
    model.jump_entry = entry;
}

static void arch_barrier(void) {
}

#define BOOT__READ32(addr_) model_read((UPtr)(addr_))
#define BOOT__WRITE32(addr_, value_) model_write((UPtr)(addr_), (U32)(value_))
#define BOOT__NOINIT
#define BOOT__SYSTEM_RESET() model_system_reset()
#define BOOT__JUMP(stack_, entry_) model_jump((stack_), (entry_))
#include "../boot.h"

static void test_request_survives_reset_and_jumps_once(void) {
    model_reset();
    boot__request = 0;

    // A cold boot with no request must not divert.
    boot_bootloader_jump_if_pending();
    CHECK(model.jumps == 0u);

    boot_request_bootloader();
    CHECK(model.resets == 1u);

    boot_bootloader_jump_if_pending();
    CHECK(model.jumps == 1u);
    CHECK(model.jump_stack == 0x20001000u);
    CHECK(model.jump_entry == 0x1FFF3421u);

    // The request is consumed, so the next reset lands in the app.
    boot_bootloader_jump_if_pending();
    CHECK(model.jumps == 1u);
}

static void test_uninitialized_ram_does_not_divert(void) {
    model_reset();

    boot__request = 0xFFFFFFFFu;
    boot_bootloader_jump_if_pending();
    CHECK(model.jumps == 0u);

    boot__request = 0x00000000u;
    boot_bootloader_jump_if_pending();
    CHECK(model.jumps == 0u);
}

static void test_implausible_vector_is_refused(void) {
    model_reset();
    boot__request = BOOT__REQUEST_MAGIC;
    model.system_memory[0] = 0x08001000u;  // stack outside SRAM
    boot_bootloader_jump_if_pending();
    CHECK(model.jumps == 0u);

    model_reset();
    boot__request = BOOT__REQUEST_MAGIC;
    model.system_memory[1] = 0x08000401u;  // entry outside system memory
    boot_bootloader_jump_if_pending();
    CHECK(model.jumps == 0u);
}

static U32 crashes_after(U32 csr) {
    model.csr = csr;
    return boot_crashes();
}

static void test_crash_accounting(void) {
    model_reset();

    // Garbage in .noinit, or a power-on, starts from zero; the flags clear.
    boot__crashes = 7u;
    boot__crashes_check = 0u;
    CHECK(crashes_after(BOOT__CSR_IWDGRSTF) == 0u);
    CHECK(model.csr == 0u);
    CHECK(crashes_after(BOOT__CSR_PWRRSTF | BOOT__CSR_IWDGRSTF) == 0u);

    // Watchdog resets count; a plain software or pin reset does not.
    CHECK(crashes_after(BOOT__CSR_IWDGRSTF) == 1u);
    CHECK(crashes_after(BOOT__CSR_WWDGRSTF) == 2u);
    CHECK(crashes_after(BOOT__CSR_SFTRSTF) == 2u);
    CHECK(crashes_after(BOOT__CSR_PINRSTF | BOOT__CSR_SFTRSTF) == 2u);
    CHECK(boot_last_reset_cause() == (BOOT__CSR_PINRSTF | BOOT__CSR_SFTRSTF));

    // A fault reset counts once, and reports its frame.
    boot_fault_reset(0x08001234u, 0x08000ABDu);
    CHECK(model.resets == 1u);
    CHECK(crashes_after(BOOT__CSR_SFTRSTF) == 3u);
    CHECK(boot_fault_pc() == 0x08001234u && boot_fault_lr() == 0x08000ABDu);
    CHECK(crashes_after(BOOT__CSR_SFTRSTF) == 3u);

    boot_mark_stable();
    CHECK(crashes_after(BOOT__CSR_SFTRSTF) == 0u);
    CHECK(crashes_after(BOOT__CSR_IWDGRSTF) == 1u);
}

static void test_flash_marked_programmed(void) {
    // After first programming: EMPTY still set from the last option byte
    // load, with latency, prefetch and debug enable configured around it.
    model_reset();
    model.acr = (1u << 18) | (1u << 16) | (1u << 8) | 2u;
    boot_mark_flash_programmed();
    CHECK(model.acr == ((1u << 18) | (1u << 8) | 2u));
    CHECK(model.acr_writes == 1u);

    // Already clear (booted after a power reset): no write.
    boot_mark_flash_programmed();
    CHECK(model.acr_writes == 1u);
}

int main(void) {
    test_flash_marked_programmed();
    test_request_survives_reset_and_jumps_once();
    test_uninitialized_ram_does_not_divert();
    test_implausible_vector_is_refused();
    test_crash_accounting();
    puts("PASS boot: bootloader handover, reset accounting, flash empty flag");
    return 0;
}
