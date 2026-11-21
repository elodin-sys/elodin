// The exported simulation, through the C interface of elodin_model.h.
#pragma once

#include "elodin_model.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

class Model {
public:
    // Opens the model and starts one run at its initial state. `workers` is the thread count of
    // the step, 0 for the count it was exported for.
    explicit Model(size_t workers);
    ~Model();
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    size_t component(const char* name) const;
    size_t rows(size_t component) const;
    std::string entity_name(size_t component, size_t row) const;
    uint64_t entity_id(size_t component, size_t row) const;

    // A component of f64 values, every row.
    std::vector<double> read(size_t component) const;
    void write(size_t component, const std::vector<double>& values);
    void step();

private:
    ElodinModelProgram* program_;
    ElodinModelInstance* instance_;
};
