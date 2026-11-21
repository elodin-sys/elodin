#include "elodin_model.hpp"

#include <cstddef>
#include <iostream>
#include <vector>

int main()
{
    try {
        elodin::Program program;
        const std::size_t agents = program.interface("agents");
        if (elodin_model_interface_rows(program.get(), agents) == 0)
            throw std::runtime_error("agents interface is empty");
        if (elodin_model_interface_field_count(program.get(), agents) != 3)
            throw std::runtime_error("agents interface field count changed");

        elodin::Instance run(program);
        const auto position = run.read_interface(agents, 0);
        const elodin::InterfaceField command = program.interface_field(agents, 2);
        if (!command.writable || position.size() != command.rows * 3 * sizeof(double))
            throw std::runtime_error("agents interface layout changed");
        std::vector<std::byte> zeros(command.rows * command.row_bytes);
        run.write_interface(agents, 2, zeros.data(), zeros.size());
        run.step();

        std::cout << "named model interface smoke passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
