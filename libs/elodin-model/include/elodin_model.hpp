#ifndef ELODIN_MODEL_HPP
#define ELODIN_MODEL_HPP

#include "elodin_model.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace elodin {

inline void check(int status, const char* operation)
{
    if (status != ELODIN_MODEL_OK)
        throw std::runtime_error(std::string(operation) + ": " + elodin_model_error());
}

struct ComponentType {
    std::string dtype;
    std::vector<std::size_t> shape;
    std::vector<std::size_t> element_shape;
    std::size_t element_bytes {};
    std::size_t rows {};
    std::size_t row_bytes {};
    std::size_t total_bytes {};
    std::uint32_t flags {};
    std::uint32_t entity_axis {};
};

struct InterfaceField {
    std::string name;
    std::size_t component {};
    bool writable {};
    std::size_t rows {};
    std::size_t row_bytes {};
};

inline std::vector<std::size_t> copy_shape(const std::size_t* shape, std::size_t rank)
{
    return rank ? std::vector<std::size_t>(shape, shape + rank) : std::vector<std::size_t> {};
}

template <typename T>
constexpr const char* dtype_name()
{
    if constexpr (std::is_same_v<T, double>)
        return "float64";
    else if constexpr (std::is_same_v<T, float>)
        return "float32";
    else if constexpr (std::is_same_v<T, std::uint64_t>)
        return "uint64";
    else if constexpr (std::is_same_v<T, std::uint32_t>)
        return "uint32";
    else if constexpr (std::is_same_v<T, std::uint16_t>)
        return "uint16";
    else if constexpr (std::is_same_v<T, std::uint8_t>)
        return "uint8";
    else if constexpr (std::is_same_v<T, std::int64_t>)
        return "int64";
    else if constexpr (std::is_same_v<T, std::int32_t>)
        return "int32";
    else if constexpr (std::is_same_v<T, std::int16_t>)
        return "int16";
    else if constexpr (std::is_same_v<T, std::int8_t>)
        return "int8";
    else if constexpr (std::is_same_v<T, bool>)
        return "bool";
    else
        return nullptr;
}

class Program {
public:
    Program()
        : handle_(elodin_model_open(), elodin_model_close)
    {
        if (!handle_)
            throw std::runtime_error(std::string("open model: ") + elodin_model_error());
    }

    ElodinModelProgram* get() const { return handle_.get(); }
    std::size_t component_count() const { return elodin_model_component_count(get()); }

    std::size_t component(const char* name) const
    {
        const std::int64_t index = elodin_model_component_index(get(), name);
        if (index < 0)
            throw std::runtime_error(std::string("no model component ") + name);
        return static_cast<std::size_t>(index);
    }

    ComponentType component_type(std::size_t index) const
    {
        ElodinModelComponentType raw {};
        raw.struct_size = sizeof raw;
        check(elodin_model_component_type(get(), index, &raw), "component type");
        return {
            raw.dtype,
            copy_shape(raw.shape, raw.rank),
            copy_shape(raw.element_shape, raw.element_rank),
            raw.element_bytes,
            raw.rows,
            raw.row_bytes,
            raw.total_bytes,
            raw.flags,
            raw.entity_axis,
        };
    }

    std::size_t interface(const char* name) const
    {
        const std::int64_t index = elodin_model_interface_index(get(), name);
        if (index < 0)
            throw std::runtime_error(std::string("no model interface ") + name);
        return static_cast<std::size_t>(index);
    }

    InterfaceField interface_field(std::size_t interface_index, std::size_t field) const
    {
        ElodinModelInterfaceField raw {};
        raw.struct_size = sizeof raw;
        check(elodin_model_interface_field(get(), interface_index, field, &raw), "interface field");
        return { raw.name, raw.component, raw.writable != 0, raw.rows, raw.row_bytes };
    }

private:
    friend class Instance;
    std::shared_ptr<ElodinModelProgram> handle_;
};

class Instance {
public:
    explicit Instance(const Program& program)
        : program_(program.handle_)
        , handle_(elodin_model_create(program_.get()), elodin_model_destroy)
    {
        if (!handle_)
            throw std::runtime_error(std::string("create model instance: ") + elodin_model_error());
    }

    ElodinModelInstance* get() const { return handle_.get(); }

    void reset() { check(elodin_model_reset_checked(get()), "reset model"); }
    void step() { check(elodin_model_step(get()), "step model"); }
    void set_time_step(double seconds)
    {
        check(elodin_model_set_time_step(get(), seconds), "set model time step");
    }

    std::uint64_t tick() const
    {
        std::uint64_t value {};
        check(elodin_model_current_tick(get(), &value), "read model tick");
        return value;
    }

    std::uint64_t generation() const { return elodin_model_generation(get()); }

    std::vector<std::byte> read(std::size_t component) const
    {
        std::vector<std::byte> bytes(elodin_model_component_size(program_.get(), component));
        check(elodin_model_read(get(), component, bytes.data(), bytes.size()), "read component");
        return bytes;
    }

    void write(std::size_t component, const void* data, std::size_t size)
    {
        check(elodin_model_write(get(), component, data, size), "write component");
    }

    template <typename T>
    std::vector<T> read_as(std::size_t component) const
    {
        static_assert(std::is_trivially_copyable_v<T>);
        const char* expected = dtype_name<T>();
        if (!expected || program_type(component).dtype != expected)
            throw std::runtime_error("component dtype does not match requested C++ type");
        const std::size_t bytes = elodin_model_component_size(program_.get(), component);
        if (bytes % sizeof(T) != 0)
            throw std::runtime_error("component size does not match requested C++ type");
        std::vector<T> values(bytes / sizeof(T));
        check(elodin_model_read(get(), component, values.data(), bytes), "read component");
        return values;
    }

    template <typename T>
    void write_as(std::size_t component, const std::vector<T>& values)
    {
        static_assert(std::is_trivially_copyable_v<T>);
        const char* expected = dtype_name<T>();
        if (!expected || program_type(component).dtype != expected)
            throw std::runtime_error("component dtype does not match requested C++ type");
        write(component, values.data(), values.size() * sizeof(T));
    }

    std::vector<std::byte> read_interface(std::size_t interface_index, std::size_t field) const
    {
        ElodinModelInterfaceField info {};
        info.struct_size = sizeof info;
        check(
            elodin_model_interface_field(program_.get(), interface_index, field, &info),
            "interface field");
        std::vector<std::byte> bytes(info.rows * info.row_bytes);
        check(
            elodin_model_interface_read(get(), interface_index, field, bytes.data(), bytes.size()),
            "read interface");
        return bytes;
    }

    void write_interface(
        std::size_t interface_index,
        std::size_t field,
        const void* data,
        std::size_t size)
    {
        check(
            elodin_model_interface_write(get(), interface_index, field, data, size),
            "write interface");
    }

private:
    ComponentType program_type(std::size_t component) const
    {
        ElodinModelComponentType raw {};
        raw.struct_size = sizeof raw;
        check(elodin_model_component_type(program_.get(), component, &raw), "component type");
        return {
            raw.dtype,
            copy_shape(raw.shape, raw.rank),
            copy_shape(raw.element_shape, raw.element_rank),
            raw.element_bytes,
            raw.rows,
            raw.row_bytes,
            raw.total_bytes,
            raw.flags,
            raw.entity_axis,
        };
    }

    std::shared_ptr<ElodinModelProgram> program_;
    std::unique_ptr<ElodinModelInstance, decltype(&elodin_model_destroy)> handle_;
};

} // namespace elodin

#endif
