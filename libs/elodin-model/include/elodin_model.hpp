// C++17 wrapper over elodin_model.h, for a class of your own to hold a model as a member. There
// is no base class to derive from, and nothing here calls back, schedules or keeps time.
//
//     elodin::Program model;                      // open once
//     elodin::Instance run(model);                // one per run; reset() starts the next case
//     const std::size_t position = model.component("position");
//     std::vector<double> xyz(3 * model.rows(position));
//     run.step();
//     run.read(position, xyz.data(), xyz.size());
//
// A failure throws std::runtime_error with the library's message. Code built without exceptions
// uses elodin_model.h, which this header only wraps; the threading and stack rules are there.
#ifndef ELODIN_MODEL_HPP
#define ELODIN_MODEL_HPP

#include "elodin_model.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace elodin {

// Throws `what` with the library's reason unless `status` is ELODIN_MODEL_OK.
inline void check(int status, const char* what)
{
    if (status != ELODIN_MODEL_OK)
        throw std::runtime_error(std::string(what) + ": " + elodin_model_error());
}

// Threads a step runs its parallel groups on; call before the first Program. 0 for the count
// the model was exported for.
inline void set_workers(std::size_t workers)
{
    check(elodin_model_set_workers(workers), "set workers");
}

// How the library was built, as JSON for a run manifest.
inline const char* build_info() { return elodin_model_build_info(); }

// The schema dtype a C++ type holds, or nullptr for a type no component has.
template <typename T>
constexpr const char* dtype_name()
{
    if constexpr (std::is_same_v<T, bool>)
        return "bool";
    else if constexpr (std::is_same_v<T, double>)
        return "float64";
    else if constexpr (std::is_same_v<T, float>)
        return "float32";
    else if constexpr (std::is_integral_v<T> && sizeof(T) == 8)
        return std::is_signed_v<T> ? "int64" : "uint64";
    else if constexpr (std::is_integral_v<T> && sizeof(T) == 4)
        return std::is_signed_v<T> ? "int32" : "uint32";
    else if constexpr (std::is_integral_v<T> && sizeof(T) == 2)
        return std::is_signed_v<T> ? "int16" : "uint16";
    else if constexpr (std::is_integral_v<T> && sizeof(T) == 1)
        return std::is_signed_v<T> ? "int8" : "uint8";
    else
        return nullptr;
}

struct ComponentType {
    std::string dtype;
    std::vector<std::size_t> shape;
    std::vector<std::size_t> element_shape;
    std::size_t element_bytes {};
    std::size_t rows {};
    std::size_t row_bytes {};
    std::size_t total_bytes {};
    std::uint32_t flags {}; // ELODIN_MODEL_COMPONENT_*
    std::uint32_t entity_axis {};
};

struct InterfaceField {
    std::string name;
    std::size_t component {};
    bool writable {};
    std::size_t rows {};
    std::size_t row_bytes {};
};

// The model's code, layout and initial state. Copies share one open program.
class Program {
public:
    Program()
        : handle_(elodin_model_open(), elodin_model_close)
    {
        if (!handle_)
            throw std::runtime_error(std::string("open model: ") + elodin_model_error());
    }

    ElodinModelProgram* get() const { return handle_.get(); }

    // Threads a step runs its parallel groups on, the caller included.
    std::size_t workers() const { return elodin_model_effective_workers(get()); }

    // Seconds of simulated time one step covers from the initial state.
    double nominal_time_step() const
    {
        double seconds {};
        check(elodin_model_nominal_time_step(get(), &seconds), "nominal time step");
        return seconds;
    }

    std::size_t component_count() const { return elodin_model_component_count(get()); }

    std::size_t component(const char* name) const
    {
        return index(elodin_model_component_index(get(), name), "component");
    }

    const char* component_name(std::size_t component_index) const
    {
        return named(elodin_model_component_name(get(), component_index), "component");
    }

    ComponentType component_type(std::size_t component_index) const
    {
        const ElodinModelComponentType raw = raw_type(component_index);
        return {
            raw.dtype,
            std::vector<std::size_t>(raw.shape, raw.shape + raw.rank),
            std::vector<std::size_t>(raw.element_shape, raw.element_shape + raw.element_rank),
            raw.element_bytes,
            raw.rows,
            raw.row_bytes,
            raw.total_bytes,
            raw.flags,
            raw.entity_axis,
        };
    }

    // Rows of a component: one per entity that has it.
    std::size_t rows(std::size_t component_index) const
    {
        return elodin_model_component_rows(get(), component_index);
    }

    std::uint64_t entity_id(std::size_t component_index, std::size_t row_index) const
    {
        return elodin_model_entity_id(get(), component_index, row_index);
    }

    const char* entity_name(std::size_t component_index, std::size_t row_index) const
    {
        return named(elodin_model_entity_name(get(), component_index, row_index), "row");
    }

    // The row of a component that belongs to the entity called `entity`.
    std::size_t row(std::size_t component_index, const char* entity) const
    {
        const std::size_t count = rows(component_index);
        for (std::size_t row_index = 0; row_index < count; ++row_index)
            if (std::strcmp(entity_name(component_index, row_index), entity) == 0)
                return row_index;
        throw std::runtime_error(std::string(component_name(component_index)) + " has no entity " + entity);
    }

    std::size_t interface_count() const { return elodin_model_interface_count(get()); }

    std::size_t interface(const char* name) const
    {
        return index(elodin_model_interface_index(get(), name), "interface");
    }

    const char* interface_name(std::size_t interface_index) const
    {
        return named(elodin_model_interface_name(get(), interface_index), "interface");
    }

    std::size_t field_count(std::size_t interface_index) const
    {
        return elodin_model_interface_field_count(get(), interface_index);
    }

    // The field of an interface called `name`.
    std::size_t field(std::size_t interface_index, const char* name) const
    {
        const std::size_t count = field_count(interface_index);
        for (std::size_t field_index = 0; field_index < count; ++field_index)
            if (std::strcmp(raw_field(interface_index, field_index).name, name) == 0)
                return field_index;
        throw std::runtime_error(std::string(interface_name(interface_index)) + " has no field " + name);
    }

    InterfaceField interface_field(std::size_t interface_index, std::size_t field_index) const
    {
        const ElodinModelInterfaceField raw = raw_field(interface_index, field_index);
        return { raw.name, raw.component, raw.writable != 0, raw.rows, raw.row_bytes };
    }

private:
    friend class Instance;

    static std::size_t index(std::int64_t found, const char* what)
    {
        if (found < 0)
            throw std::runtime_error(std::string("find ") + what + ": " + elodin_model_error());
        return static_cast<std::size_t>(found);
    }

    static const char* named(const char* name, const char* what)
    {
        if (!name)
            throw std::out_of_range(std::string("no such ") + what);
        return name;
    }

    ElodinModelComponentType raw_type(std::size_t component_index) const
    {
        ElodinModelComponentType raw {};
        raw.struct_size = sizeof raw;
        check(elodin_model_component_type(get(), component_index, &raw), "component type");
        return raw;
    }

    ElodinModelInterfaceField raw_field(std::size_t interface_index, std::size_t field_index) const
    {
        ElodinModelInterfaceField raw {};
        raw.struct_size = sizeof raw;
        check(elodin_model_interface_field(get(), interface_index, field_index, &raw), "interface field");
        return raw;
    }

    // Throws unless the component holds `T`. `std::byte` stands for raw bytes of any type.
    template <typename T>
    void expect(std::size_t component_index) const
    {
        static_assert(sizeof(bool) == 1, "bool components are one byte");
        if constexpr (!std::is_same_v<T, std::byte>) {
            static_assert(dtype_name<T>() != nullptr, "use a component's value type, or std::byte");
            const ElodinModelComponentType raw = raw_type(component_index);
            if (std::strcmp(raw.dtype, dtype_name<T>()) != 0)
                throw std::runtime_error(std::string(component_name(component_index)) + " holds "
                    + raw.dtype + ", not " + dtype_name<T>());
        }
    }

    std::shared_ptr<ElodinModelProgram> handle_;
};

// The state of one run. It keeps its program open. In the typed calls below, `count` is the
// number of `T` values at `values`, and `T` must be the component's type or `std::byte`; they
// copy, and allocate nothing.
class Instance {
public:
    explicit Instance(const Program& program)
        : program_(program)
        , handle_(elodin_model_create(program_.get()), elodin_model_destroy)
    {
        if (!handle_)
            throw std::runtime_error(std::string("create model instance: ") + elodin_model_error());
    }

    ElodinModelInstance* get() const { return handle_.get(); }

    // Back to the initial state; write the next case's inputs after it.
    void reset() { check(elodin_model_reset(get()), "reset model"); }
    // One tick, on the calling thread.
    void step() { check(elodin_model_step(get()), "step model"); }
    // For a model with a time-step component; others keep the step they were built with.
    void set_time_step(double seconds)
    {
        check(elodin_model_set_time_step(get(), seconds), "set model time step");
    }

    // For a model with a tick component.
    std::uint64_t tick() const
    {
        std::uint64_t value {};
        check(elodin_model_current_tick(get(), &value), "read model tick");
        return value;
    }

    std::uint64_t generation() const { return elodin_model_generation(get()); }

    // Every row of a component.
    template <typename T>
    void read(std::size_t component_index, T* values, std::size_t count) const
    {
        program_.expect<T>(component_index);
        check(elodin_model_read(get(), component_index, values, count * sizeof(T)), "read component");
    }

    template <typename T>
    void write(std::size_t component_index, const T* values, std::size_t count)
    {
        program_.expect<T>(component_index);
        check(elodin_model_write(get(), component_index, values, count * sizeof(T)), "write component");
    }

    template <typename T>
    std::vector<T> read(std::size_t component_index) const
    {
        static_assert(!std::is_same_v<T, bool>, "std::vector<bool> is packed; read into a bool array");
        std::vector<T> values(elodin_model_component_size(program_.get(), component_index) / sizeof(T));
        read(component_index, values.data(), values.size());
        return values;
    }

    template <typename T>
    void write(std::size_t component_index, const std::vector<T>& values)
    {
        static_assert(!std::is_same_v<T, bool>, "std::vector<bool> is packed; write from a bool array");
        write(component_index, values.data(), values.size());
    }

    // `row_count` rows of a component, from row `first`.
    template <typename T>
    void read_rows(std::size_t component_index, std::size_t first, std::size_t row_count, T* values,
        std::size_t count) const
    {
        program_.expect<T>(component_index);
        check(
            elodin_model_read_rows(get(), component_index, first, row_count, values, count * sizeof(T)),
            "read rows");
    }

    template <typename T>
    void write_rows(std::size_t component_index, std::size_t first, std::size_t row_count, const T* values,
        std::size_t count)
    {
        program_.expect<T>(component_index);
        check(
            elodin_model_write_rows(get(), component_index, first, row_count, values, count * sizeof(T)),
            "write rows");
    }

    // Every row of one field of an interface, in the interface's row order.
    template <typename T>
    void read_interface(std::size_t interface_index, std::size_t field_index, T* values, std::size_t count) const
    {
        expect_field<T>(interface_index, field_index);
        check(
            elodin_model_interface_read(get(), interface_index, field_index, values, count * sizeof(T)),
            "read interface");
    }

    template <typename T>
    void write_interface(std::size_t interface_index, std::size_t field_index, const T* values,
        std::size_t count)
    {
        expect_field<T>(interface_index, field_index);
        check(
            elodin_model_interface_write(get(), interface_index, field_index, values, count * sizeof(T)),
            "write interface");
    }

    template <typename T>
    std::vector<T> read_interface(std::size_t interface_index, std::size_t field_index) const
    {
        static_assert(!std::is_same_v<T, bool>, "std::vector<bool> is packed; read into a bool array");
        const ElodinModelInterfaceField raw = program_.raw_field(interface_index, field_index);
        std::vector<T> values(raw.rows * raw.row_bytes / sizeof(T));
        read_interface(interface_index, field_index, values.data(), values.size());
        return values;
    }

    template <typename T>
    void write_interface(std::size_t interface_index, std::size_t field_index, const std::vector<T>& values)
    {
        static_assert(!std::is_same_v<T, bool>, "std::vector<bool> is packed; write from a bool array");
        write_interface(interface_index, field_index, values.data(), values.size());
    }

private:
    template <typename T>
    void expect_field(std::size_t interface_index, std::size_t field_index) const
    {
        if constexpr (!std::is_same_v<T, std::byte>)
            program_.expect<T>(program_.raw_field(interface_index, field_index).component);
    }

    Program program_;
    std::unique_ptr<ElodinModelInstance, decltype(&elodin_model_destroy)> handle_;
};

} // namespace elodin

#endif
