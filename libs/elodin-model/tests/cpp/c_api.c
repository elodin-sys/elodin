/* Every function of elodin_model.h, from C11, against the rooms model of tests/fixtures/rooms:
 * each step moves each room's temperature toward its ambient temperature by the time step. */
#include "elodin_model.h"
#include "elodin_step_hook.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum { TICK, TEMPERATURE, AMBIENT, TIME_STEP, COMPONENTS };
enum { MEASURED, COMMAND, FIELDS };

static int failures;

#define CHECK(condition)                                                                       \
    do {                                                                                       \
        if (!(condition)) {                                                                    \
            fprintf(stderr, "%s:%d: %s (%s)\n", __FILE__, __LINE__, #condition,                \
                    elodin_model_error());                                                     \
            ++failures;                                                                        \
        }                                                                                      \
    } while (0)

static int pair_is(const double values[2], double first, double second)
{
    return values[0] == first && values[1] == second;
}

static void describes_the_library(void)
{
    CHECK(elodin_model_abi_version() == ELODIN_MODEL_ABI_VERSION);
    CHECK(strcmp(elodin_model_backend(), "aot") == 0 || strcmp(elodin_model_backend(), "jit") == 0);
    CHECK(strstr(elodin_model_build_info(), "\"elodin.model-build\"") != NULL);
    CHECK(strstr(elodin_model_build_info(), "\"compiled_at_utc\"") != NULL);
    CHECK(elodin_model_recommended_stack() >= (size_t)1 << 20);
    CHECK(elodin_model_error()[0] == '\0');
}

static void describes_the_model(const ElodinModelProgram *program)
{
    const int aot = strcmp(elodin_model_backend(), "aot") == 0;
    CHECK(elodin_model_schema_abi(program) == ELODIN_MODEL_ABI_VERSION);
    CHECK(strcmp(elodin_model_format(program), "elodin.model") == 0);
    CHECK(strcmp(elodin_model_endianness(program), "little") == 0);
    /* The rooms were exported for two workers; the jit backend always steps on the caller. */
    CHECK(elodin_model_effective_workers(program) == (aot ? 2u : 1u));
    CHECK(elodin_model_tick_advanced_by_step(program) == 1);
    double seconds = 0.0;
    CHECK(elodin_model_nominal_time_step(program, &seconds) == ELODIN_MODEL_OK && seconds == 0.25);

    CHECK(elodin_model_component_count(program) == COMPONENTS);
    CHECK(strcmp(elodin_model_component_name(program, AMBIENT), "ambient") == 0);
    CHECK(elodin_model_component_name(program, COMPONENTS) == NULL);
    CHECK(elodin_model_component_index(program, "temperature") == TEMPERATURE);
    CHECK(elodin_model_component_size(program, TEMPERATURE) == 16);
    CHECK(elodin_model_component_size(program, COMPONENTS) == 0);
    CHECK(elodin_model_component_rows(program, TEMPERATURE) == 2);
    CHECK(elodin_model_component_rows(program, TICK) == 1);
    CHECK(strcmp(elodin_model_entity_name(program, TEMPERATURE, 1), "room_1") == 0);
    CHECK(elodin_model_entity_name(program, TEMPERATURE, 2) == NULL);
    CHECK(elodin_model_entity_id(program, TEMPERATURE, 1) == 2);
    CHECK(elodin_model_entity_id(program, TEMPERATURE, 2) == UINT64_MAX);

    ElodinModelComponentType type = { .struct_size = sizeof type };
    CHECK(elodin_model_component_type(program, TEMPERATURE, &type) == ELODIN_MODEL_OK);
    CHECK(strcmp(type.dtype, "float64") == 0 && type.element_bytes == 8);
    CHECK(type.rank == 1 && type.shape[0] == 2 && type.element_rank == 0 && type.entity_axis == 1);
    CHECK(type.rows == 2 && type.row_bytes == 8 && type.total_bytes == 16);
    CHECK(type.flags == ELODIN_MODEL_COMPONENT_WRITTEN_BY_STEP);
    const uint32_t flags[COMPONENTS] = {
        ELODIN_MODEL_COMPONENT_WRITTEN_BY_STEP | ELODIN_MODEL_COMPONENT_TICK,
        ELODIN_MODEL_COMPONENT_WRITTEN_BY_STEP,
        ELODIN_MODEL_COMPONENT_EXTERNAL_CONTROL,
        ELODIN_MODEL_COMPONENT_TIME_STEP,
    };
    for (size_t component = 0; component < COMPONENTS; ++component) {
        CHECK(elodin_model_component_type(program, component, &type) == ELODIN_MODEL_OK);
        CHECK(type.flags == flags[component]);
    }
    CHECK(type.rank == 0 && type.entity_axis == 0 && type.rows == 1);

    CHECK(elodin_model_interface_count(program) == 1);
    CHECK(strcmp(elodin_model_interface_name(program, 0), "rooms.exchange") == 0);
    CHECK(elodin_model_interface_name(program, 1) == NULL);
    CHECK(elodin_model_interface_index(program, "rooms.exchange") == 0);
    CHECK(elodin_model_interface_rows(program, 0) == 2);
    CHECK(elodin_model_interface_field_count(program, 0) == FIELDS);
    ElodinModelInterfaceField field = { .struct_size = sizeof field };
    CHECK(elodin_model_interface_field(program, 0, MEASURED, &field) == ELODIN_MODEL_OK);
    CHECK(strcmp(field.name, "measured") == 0 && field.component == TEMPERATURE && !field.writable);
    CHECK(elodin_model_interface_field(program, 0, COMMAND, &field) == ELODIN_MODEL_OK);
    CHECK(strcmp(field.name, "command") == 0 && field.component == AMBIENT && field.writable);
    CHECK(field.rows == 2 && field.row_bytes == 8);
}

static void steps_reads_and_writes(const ElodinModelProgram *program)
{
    ElodinModelInstance *run = elodin_model_create(program);
    CHECK(run != NULL);
    CHECK(elodin_model_generation(run) == 0);
    double temperature[2], ambient[2] = { 30.0, 22.0 }, row, seconds;
    uint64_t tick = 99;
    CHECK(elodin_model_current_tick(run, &tick) == ELODIN_MODEL_OK && tick == 0);
    CHECK(elodin_model_read(run, TEMPERATURE, temperature, sizeof temperature) == ELODIN_MODEL_OK);
    CHECK(pair_is(temperature, 20.0, 28.0));

    CHECK(elodin_model_step(run) == ELODIN_MODEL_OK);
    CHECK(elodin_model_read(run, TEMPERATURE, temperature, sizeof temperature) == ELODIN_MODEL_OK);
    CHECK(pair_is(temperature, 22.0, 26.0));
    CHECK(elodin_model_current_tick(run, &tick) == ELODIN_MODEL_OK && tick == 1);
    CHECK(elodin_model_generation(run) == 1);

    /* A write is seen by the next step. */
    CHECK(elodin_model_write(run, AMBIENT, ambient, sizeof ambient) == ELODIN_MODEL_OK);
    CHECK(elodin_model_generation(run) == 2);
    CHECK(elodin_model_step(run) == ELODIN_MODEL_OK);
    CHECK(elodin_model_read(run, TEMPERATURE, temperature, sizeof temperature) == ELODIN_MODEL_OK);
    CHECK(pair_is(temperature, 24.0, 25.0));

    /* Rows. */
    CHECK(elodin_model_read_rows(run, TEMPERATURE, 1, 1, &row, sizeof row) == ELODIN_MODEL_OK);
    CHECK(row == 25.0);
    row = 18.0;
    CHECK(elodin_model_write_rows(run, AMBIENT, 0, 1, &row, sizeof row) == ELODIN_MODEL_OK);
    CHECK(elodin_model_read(run, AMBIENT, ambient, sizeof ambient) == ELODIN_MODEL_OK);
    CHECK(pair_is(ambient, 18.0, 22.0));
    CHECK(elodin_model_read_rows(run, TEMPERATURE, 0, 2, temperature, sizeof temperature) == ELODIN_MODEL_OK);
    CHECK(pair_is(temperature, 24.0, 25.0));
    CHECK(elodin_model_read_rows(run, TEMPERATURE, 2, 0, NULL, 0) == ELODIN_MODEL_OK);

    /* Reset restores every component, the tick and time step included. */
    CHECK(elodin_model_set_time_step(run, 0.5) == ELODIN_MODEL_OK);
    CHECK(elodin_model_reset(run) == ELODIN_MODEL_OK);
    CHECK(elodin_model_current_tick(run, &tick) == ELODIN_MODEL_OK && tick == 0);
    CHECK(elodin_model_read(run, TIME_STEP, &seconds, sizeof seconds) == ELODIN_MODEL_OK);
    CHECK(seconds == 0.25);
    CHECK(elodin_model_read(run, AMBIENT, ambient, sizeof ambient) == ELODIN_MODEL_OK);
    CHECK(pair_is(ambient, 28.0, 20.0));

    /* The caller's time step, for a model that has a time-step component. */
    CHECK(elodin_model_set_time_step(run, 0.5) == ELODIN_MODEL_OK);
    CHECK(elodin_model_step(run) == ELODIN_MODEL_OK);
    CHECK(elodin_model_read(run, TEMPERATURE, temperature, sizeof temperature) == ELODIN_MODEL_OK);
    CHECK(pair_is(temperature, 24.0, 24.0));
    CHECK(elodin_model_set_time_step(run, 0.0) == ELODIN_MODEL_ERR_ARGUMENT);
    CHECK(elodin_model_set_time_step(run, -1.0) == ELODIN_MODEL_ERR_ARGUMENT);

    /* An interface gathers rows in its own order: `measured` lists room_1 first. */
    CHECK(elodin_model_reset(run) == ELODIN_MODEL_OK);
    double measured[2], command[2] = { 30.0, 22.0 };
    CHECK(elodin_model_interface_read(run, 0, MEASURED, measured, sizeof measured) == ELODIN_MODEL_OK);
    CHECK(pair_is(measured, 28.0, 20.0));
    CHECK(elodin_model_interface_write(run, 0, COMMAND, command, sizeof command) == ELODIN_MODEL_OK);
    CHECK(elodin_model_step(run) == ELODIN_MODEL_OK);
    CHECK(elodin_model_interface_read(run, 0, MEASURED, measured, sizeof measured) == ELODIN_MODEL_OK);
    CHECK(pair_is(measured, 26.5, 22.5));
    CHECK(elodin_model_interface_write(run, 0, MEASURED, command, sizeof command) == ELODIN_MODEL_ERR_ARGUMENT);
    CHECK(strstr(elodin_model_error(), "read-only") != NULL);

    /* A second run is independent of the first. */
    ElodinModelInstance *other = elodin_model_create(program);
    CHECK(elodin_model_read(other, TEMPERATURE, temperature, sizeof temperature) == ELODIN_MODEL_OK);
    CHECK(pair_is(temperature, 20.0, 28.0));
    elodin_model_destroy(other);
    elodin_model_destroy(run);
    elodin_model_destroy(NULL);
}

static void reports_failures(const ElodinModelProgram *program)
{
    ElodinModelInstance *run = elodin_model_create(program);
    double values[4] = { 0 };
    CHECK(elodin_model_read(run, COMPONENTS, values, 8) == ELODIN_MODEL_ERR_ARGUMENT);
    const char *message = elodin_model_error();
    CHECK(strcmp(message, "no component 4") == 0);
    /* The message outlives later successes; only another failure replaces it. */
    CHECK(elodin_model_step(run) == ELODIN_MODEL_OK);
    CHECK(elodin_model_read(run, TEMPERATURE, values, 16) == ELODIN_MODEL_OK);
    CHECK(elodin_model_error() == message && strcmp(message, "no component 4") == 0);

    CHECK(elodin_model_read(run, TEMPERATURE, values, 8) == ELODIN_MODEL_ERR_ARGUMENT);
    CHECK(elodin_model_write(run, TEMPERATURE, values, 32) == ELODIN_MODEL_ERR_ARGUMENT);
    CHECK(elodin_model_read(run, TEMPERATURE, NULL, 16) == ELODIN_MODEL_ERR_ARGUMENT);
    CHECK(elodin_model_read_rows(run, TEMPERATURE, 1, 2, values, 16) == ELODIN_MODEL_ERR_ARGUMENT);
    CHECK(elodin_model_read_rows(run, TEMPERATURE, SIZE_MAX, 1, values, 8) == ELODIN_MODEL_ERR_ARGUMENT);
    CHECK(elodin_model_write_rows(run, TEMPERATURE, 0, 1, values, 16) == ELODIN_MODEL_ERR_ARGUMENT);
    CHECK(elodin_model_interface_read(run, 1, 0, values, 16) == ELODIN_MODEL_ERR_ARGUMENT);
    CHECK(elodin_model_interface_read(run, 0, FIELDS, values, 16) == ELODIN_MODEL_ERR_ARGUMENT);
    CHECK(elodin_model_interface_write(run, 0, COMMAND, values, 8) == ELODIN_MODEL_ERR_ARGUMENT);

    CHECK(elodin_model_component_index(program, "humidity") == -1);
    CHECK(strcmp(elodin_model_error(), "no component named humidity") == 0);
    CHECK(elodin_model_interface_index(program, "attic") == -1);
    CHECK(strcmp(elodin_model_error(), "no interface named attic") == 0);
    ElodinModelComponentType type = { .struct_size = sizeof type - 1 };
    CHECK(elodin_model_component_type(program, TICK, &type) == ELODIN_MODEL_ERR_ARGUMENT);
    ElodinModelInterfaceField field = { .struct_size = 0 };
    CHECK(elodin_model_interface_field(program, 0, COMMAND, &field) == ELODIN_MODEL_ERR_ARGUMENT);

    /* NULL handles fail; they do not crash. */
    uint64_t tick;
    CHECK(elodin_model_step(NULL) == ELODIN_MODEL_ERR_ARGUMENT);
    CHECK(elodin_model_reset(NULL) == ELODIN_MODEL_ERR_ARGUMENT);
    CHECK(elodin_model_current_tick(NULL, &tick) == ELODIN_MODEL_ERR_ARGUMENT);
    CHECK(elodin_model_current_tick(run, NULL) == ELODIN_MODEL_ERR_ARGUMENT);
    CHECK(elodin_model_create(NULL) == NULL);
    CHECK(elodin_model_component_count(NULL) == 0 && elodin_model_interface_count(NULL) == 0);
    CHECK(elodin_model_generation(NULL) == 0 && elodin_model_schema_abi(NULL) == 0);
    CHECK(elodin_model_component_index(NULL, "tick") == -1);
    CHECK(elodin_model_component_index(program, NULL) == -1);
    elodin_model_destroy(run);
}

int main(void)
{
    describes_the_library();
    CHECK(elodin_model_set_workers(0) == ELODIN_MODEL_OK);
    ElodinModelProgram *program = elodin_model_open();
    if (!program) {
        fprintf(stderr, "open: %s\n", elodin_model_error());
        return 1;
    }
    /* The worker count is fixed once a program is open. */
    CHECK(elodin_model_set_workers(1) == ELODIN_MODEL_ERR_STATE);
    describes_the_model(program);
    steps_reads_and_writes(program);
    reports_failures(program);
    elodin_model_close(program);
    elodin_model_close(NULL);
    CHECK(elodin_model_set_workers(0) == ELODIN_MODEL_OK);
    if (failures == 0)
        puts("the C interface matches the rooms model");
    return failures != 0;
}
