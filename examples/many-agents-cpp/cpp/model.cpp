#include "model.hpp"

#include <stdexcept>

namespace {

[[noreturn]] void fail(const std::string& what)
{
    throw std::runtime_error(what + ": " + elodin_model_error());
}

} // namespace

Model::Model(size_t workers)
{
    elodin_model_set_workers(workers);
    program_ = elodin_model_open();
    if (!program_)
        fail("open the model");
    instance_ = elodin_model_create(program_);
    if (!instance_) {
        elodin_model_close(program_);
        fail("create a run");
    }
}

Model::~Model()
{
    elodin_model_destroy(instance_);
    elodin_model_close(program_);
}

size_t Model::component(const char* name) const
{
    const int64_t index = elodin_model_component_index(program_, name);
    if (index < 0)
        throw std::runtime_error(std::string("the model has no component ") + name);
    return static_cast<size_t>(index);
}

size_t Model::rows(size_t component) const { return elodin_model_component_rows(program_, component); }

std::string Model::entity_name(size_t component, size_t row) const
{
    const char* name = elodin_model_entity_name(program_, component, row);
    return name ? name : "";
}

uint64_t Model::entity_id(size_t component, size_t row) const
{
    return elodin_model_entity_id(program_, component, row);
}

std::vector<double> Model::read(size_t component) const
{
    std::vector<double> values(elodin_model_component_size(program_, component) / sizeof(double));
    if (elodin_model_read(instance_, component, values.data(), values.size() * sizeof(double)))
        fail("read a component");
    return values;
}

void Model::write(size_t component, const std::vector<double>& values)
{
    if (elodin_model_write(instance_, component, values.data(), values.size() * sizeof(double)))
        fail("write a component");
}

void Model::step()
{
    if (elodin_model_step(instance_))
        fail("step");
}
