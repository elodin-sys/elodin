#ifndef ELO_DB_VTABLE_H
#define ELO_DB_VTABLE_H

#include <string_view>
#if __has_include("db.hpp")
#include "db.hpp"
#endif

#include <cstddef>
#include <cstdint>
#include <memory>
#include <print>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <variant>
#include <vector>

namespace vtable {

/// A builder for VTable operations
class OpBuilder {
public:
    struct Data {
        size_t align;
        std::vector<uint8_t> data;
    };

    struct Table {
        uint16_t offset;
        uint16_t len;
    };

    struct Component {
        std::shared_ptr<OpBuilder> component_id;
    };

    struct Schema {
        std::shared_ptr<OpBuilder> ty;
        std::shared_ptr<OpBuilder> dim;
        std::shared_ptr<OpBuilder> arg;
    };

    struct Timestamp {
        std::shared_ptr<OpBuilder> timestamp;
        std::shared_ptr<OpBuilder> arg;
    };

    struct Ext {
        std::tuple<uint8_t, uint8_t> id;
        std::shared_ptr<OpBuilder> data;
        std::shared_ptr<OpBuilder> arg;
    };

    using ValueType = std::variant<Data, Table, Component, Schema, Timestamp, Ext>;
    ValueType value;

    explicit OpBuilder(const Data& data)
        : value(data)
    {
    }

    explicit OpBuilder(const Table& table)
        : value(table)
    {
    }

    explicit OpBuilder(const Component& component)
        : value(component)
    {
    }

    explicit OpBuilder(const Schema& schema)
        : value(schema)
    {
    }

    explicit OpBuilder(const Timestamp& timestamp)
        : value(timestamp)
    {
    }

    explicit OpBuilder(const Ext& ext)
        : value(ext)
    {
    }

    // Prevent copy, only allow move
    OpBuilder(const OpBuilder&) = delete;
    OpBuilder& operator=(const OpBuilder&) = delete;

    OpBuilder(OpBuilder&& other) noexcept = default;
    OpBuilder& operator=(OpBuilder&& other) noexcept = default;
};

/// A builder for VTable fields
class FieldBuilder {
public:
    uint16_t offset;
    uint16_t len;
    std::shared_ptr<OpBuilder> arg;

    FieldBuilder(uint16_t offset, uint16_t len, std::shared_ptr<OpBuilder> arg)
        : offset(offset)
        , len(len)
        , arg(std::move(arg))
    {
    }
};

namespace builder {

    /// Create an OpBuilder that includes the passed in data in the data section
    /// ## Safety
    /// The type passed here should be safe to memcpy
    template <typename T>
    std::shared_ptr<OpBuilder> data(const std::vector<T>& values)
    {
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(values.data());
        std::vector<uint8_t> data_vec(bytes, bytes + sizeof(T) * values.size());
        OpBuilder::Data data { alignof(T), std::move(data_vec) };
        return std::make_shared<OpBuilder>(data);
    }

    /// Create an OpBuilder that includes the passed in data in the data section
    /// ## Safety
    /// The type passed here should be safe to memcpy
    template <typename T>
    std::shared_ptr<OpBuilder> data(const T& value)
    {
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&value);
        std::vector<uint8_t> data_vec(bytes, bytes + sizeof(T));
        OpBuilder::Data data { alignof(T), std::move(data_vec) };
        return std::make_shared<OpBuilder>(data);
    }

    /// Create an OpBuilder that includes the passed in data in the data section,
    /// this includes an optional alignment value that
    inline std::shared_ptr<OpBuilder> data(const std::vector<uint8_t>& bytes, size_t align = 1)
    {
        OpBuilder::Data data { align, bytes };
        return std::make_shared<OpBuilder>(data);
    }

    /// Creates an OpBuilder with the table specified offset and length
    inline std::shared_ptr<OpBuilder> raw_table(uint16_t offset, uint16_t len)
    {
        OpBuilder::Table table { offset, len };
        return std::make_shared<OpBuilder>(table);
    }

    /// Creates a operation builder from a component ID
    inline std::shared_ptr<OpBuilder> component(std::string_view component_name)
    {
        auto id = component_id(component_name);
        auto component_id_op = data(id);

        OpBuilder::Component component { std::move(component_id_op) };
        return std::make_shared<OpBuilder>(component);
    }

    /// Creates a schema operation builder from a primitive type, dimensions, and an argument
    inline std::shared_ptr<OpBuilder> schema(PrimType ty, const std::vector<uint64_t>& dim, std::shared_ptr<OpBuilder> arg)
    {
        auto ty_op = builder::data(static_cast<uint64_t>(ty.index()));
        auto dim_op = builder::data(dim);

        OpBuilder::Schema schema { std::move(ty_op), std::move(dim_op), std::move(arg) };
        return std::make_shared<OpBuilder>(schema);
    }

    /// Creates a timestamp operation builder from a timestamp source and an argument
    inline std::shared_ptr<OpBuilder> timestamp(std::shared_ptr<OpBuilder> timestamp, std::shared_ptr<OpBuilder> arg)
    {
        OpBuilder::Timestamp ts { std::move(timestamp), std::move(arg) };
        return std::make_shared<OpBuilder>(ts);
    }

    /**
     * Creates an extension operation builder from a message ID, data, and an argument
     */
    inline std::shared_ptr<OpBuilder> ext(std::tuple<uint8_t, uint8_t> id, std::shared_ptr<OpBuilder> data,
        std::shared_ptr<OpBuilder> arg)
    {
        OpBuilder::Ext ext { id, std::move(data), std::move(arg) };
        return std::make_shared<OpBuilder>(ext);
    }

    /// Creates a field builder with the specified offset, length, and argument
    inline FieldBuilder raw_field(uint16_t offset, uint16_t len, std::shared_ptr<OpBuilder> arg)
    {
        return FieldBuilder(offset, len, std::move(arg));
    }

    /// Creates a field builder from a class and its field
    /// ## Usage
    /// ```cpp
    ///  builder::field<Foo, &Foo::time>(builder::schema(PrimType::F64(), {}, builder::component("time"))),
    /// ```
    template <typename Class, auto MemberPtr>
    inline FieldBuilder field(std::shared_ptr<OpBuilder> arg)
    {
        using FieldType = std::remove_reference_t<decltype(std::declval<Class>().*MemberPtr)>;
        size_t offset = reinterpret_cast<size_t>(&(reinterpret_cast<Class*>(0)->*MemberPtr));
        size_t size = sizeof(FieldType);
        return raw_field(static_cast<uint16_t>(offset), static_cast<uint16_t>(size), std::move(arg));
    }

    /// A builder for constructing VTables
    class VTableBuilder {
    private:
        VTable vtable;
        std::unordered_map<const OpBuilder*, OpRef> visited;

    public:
        VTableBuilder() = default;

        /// Visits an operation builder, adding it to the VTable and returning its OpRef
        OpRef visit(const std::shared_ptr<OpBuilder>& op)
        {
            auto it = visited.find(op.get());
            if (it != visited.end()) {
                return it->second;
            }
            Op result_op;

            std::visit([&](const auto& val) {
                using T = std::decay_t<decltype(val)>;

                if constexpr (std::is_same_v<T, OpBuilder::Data>) {
                    const auto& data_op = val;

                    const size_t align = data_op.align;
                    const size_t padding = (align - (vtable.data.size() % align)) % align;

                    for (size_t i = 0; i < padding; i++) {
                        vtable.data.push_back(0);
                    }

                    const uint16_t offset = static_cast<uint16_t>(vtable.data.size());
                    const uint16_t len = static_cast<uint16_t>(data_op.data.size());

                    vtable.data.insert(vtable.data.end(), data_op.data.begin(), data_op.data.end());

                    result_op = Op::Data(OpData { offset, len });

                } else if constexpr (std::is_same_v<T, OpBuilder::Table>) {
                    const auto& table_op = val;
                    result_op = Op::Table(OpTable { table_op.offset, table_op.len });
                } else if constexpr (std::is_same_v<T, OpBuilder::Component>) {
                    const auto& component_op = val;
                    OpRef component_id = visit(component_op.component_id);

                    result_op = Op::Component(OpComponent {
                        static_cast<uint16_t>(component_id.value) });
                } else if constexpr (std::is_same_v<T, OpBuilder::Schema>) {
                    const auto& schema_op = val;
                    OpRef ty = visit(schema_op.ty);
                    OpRef dim = visit(schema_op.dim);
                    OpRef arg = visit(schema_op.arg);

                    result_op = Op::Schema(OpSchema {
                        static_cast<uint16_t>(ty.value),
                        static_cast<uint16_t>(dim.value),
                        static_cast<uint16_t>(arg.value) });
                } else if constexpr (std::is_same_v<T, OpBuilder::Timestamp>) {
                    const auto& timestamp_op = val;
                    OpRef timestamp = visit(timestamp_op.timestamp);
                    OpRef arg = visit(timestamp_op.arg);

                    result_op = Op::Timestamp(OpTimestamp {
                        static_cast<uint16_t>(timestamp.value),
                        static_cast<uint16_t>(arg.value) });
                } else if constexpr (std::is_same_v<T, OpBuilder::Ext>) {
                    const auto& ext_op = val;
                    OpRef data = visit(ext_op.data);
                    OpRef arg = visit(ext_op.arg);

                    result_op = Op::Ext(OpExt {
                        static_cast<uint16_t>(arg.value),
                        ext_op.id,
                        static_cast<uint16_t>(data.value) });
                }
            },
                op->value);

            OpRef op_ref(static_cast<uint16_t>(vtable.ops.size()));
            vtable.ops.push_back(result_op);
            visited[op.get()] = op_ref;

            return op_ref;
        }

        /// Adds a field to the VTable
        void push_field(const FieldBuilder& field_builder)
        {
            OpRef arg = visit(field_builder.arg);

            Field field {
                field_builder.offset,
                field_builder.len,
                static_cast<uint16_t>(arg.value)
            };

            vtable.fields.push_back(field);
        }

        /// Builds and returns the VTable
        VTable build() const
        {
            return vtable;
        }
    };

    /// Creates a VTable from the provided field builders
    inline VTable vtable(const std::initializer_list<FieldBuilder> fields)
    {
        VTableBuilder builder;

        for (const auto& field : fields) {
            builder.push_field(field);
        }

        return builder.build();
    }

// Convenience macro to create a table operation for a struct field
#define TABLE(type, field)                            \
    vtable::builder::raw_table(                       \
        static_cast<uint16_t>(offsetof(type, field)), \
        static_cast<uint16_t>(sizeof(((type*)nullptr)->field)))

// Convenience macro to create a field builder for a struct field
#define FIELD(type, field, arg)                                 \
    vtable::builder::raw_field(                                 \
        static_cast<uint16_t>(offsetof(type, field)),           \
        static_cast<uint16_t>(sizeof(((type*)nullptr)->field)), \
        arg)

} // namespace builder
} // namespace vtable

#endif // ELO_DB_VTABLE_H
