#include "exchange.hpp"

#include "protocol.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstring>
#include <stdexcept>
#include <thread>

extern char** environ;

namespace {

sockaddr_in localhost(uint16_t port)
{
    sockaddr_in address {};
    address.sin_family = AF_INET;
    address.sin_port = htons(port);
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    return address;
}

[[noreturn]] void fail(const std::string& what)
{
    throw std::runtime_error(what + ": " + std::strerror(errno));
}

} // namespace

InProcessExchange::InProcessExchange(size_t agents)
{
    for (size_t agent = 0; agent < agents; ++agent)
        agents_.emplace_back(agent);
}

void InProcessExchange::run(const uint8_t* states, uint8_t* replies)
{
    for (size_t agent = 0; agent < agents_.size(); ++agent) {
        const auto started = std::chrono::steady_clock::now();
        uint8_t* reply = replies + agent * protocol::COMMAND_SIZE;
        if (!agents_[agent].step(states + agent * protocol::STATE_SIZE, protocol::STATE_SIZE, reply))
            throw std::runtime_error("the flight software rejected a state packet");
        const uint64_t service_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started)
                .count());
        std::memcpy(reply + protocol::COMMAND_SIZE - 8, &service_ns, 8);
    }
}

ProcessExchange::ProcessExchange(const std::string& binary, size_t agents, uint16_t port_base,
    std::chrono::milliseconds timeout)
    : agents_(agents)
    , port_base_(port_base)
    , timeout_(timeout)
{
    sender_ = socket(AF_INET, SOCK_DGRAM, 0);
    if (sender_ < 0)
        fail("socket");
    std::vector<int> ready_pipes;
    try {
        for (size_t agent = 0; agent < agents; ++agent) {
            // Bound before the process starts, so no reply can arrive unheard.
            const int receiver = socket(AF_INET, SOCK_DGRAM, 0);
            if (receiver < 0)
                fail("socket");
            receivers_.push_back(receiver);
            const sockaddr_in address = localhost(static_cast<uint16_t>(port_base + 2 * agent + 1));
            if (bind(receiver, reinterpret_cast<const sockaddr*>(&address), sizeof address) != 0)
                fail("bind the reply port of agent " + std::to_string(agent));

            int out[2];
            if (pipe(out) != 0)
                fail("pipe");
            posix_spawn_file_actions_t actions;
            posix_spawn_file_actions_init(&actions);
            posix_spawn_file_actions_adddup2(&actions, out[1], STDOUT_FILENO);
            posix_spawn_file_actions_addclose(&actions, out[0]);
            const std::string state_port = std::to_string(port_base + 2 * agent);
            const std::string command_port = std::to_string(port_base + 2 * agent + 1);
            const std::string index = std::to_string(agent);
            char* argv[] = { const_cast<char*>(binary.c_str()), const_cast<char*>(state_port.c_str()),
                const_cast<char*>(command_port.c_str()), const_cast<char*>(index.c_str()), nullptr };
            pid_t pid;
            const int spawned = posix_spawn(&pid, binary.c_str(), &actions, nullptr, argv, environ);
            posix_spawn_file_actions_destroy(&actions);
            close(out[1]);
            if (spawned != 0) {
                close(out[0]);
                errno = spawned;
                fail("start " + binary);
            }
            children_.push_back(pid);
            ready_pipes.push_back(out[0]);
        }
        // Each process says `ready` once its state port is bound.
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        for (size_t agent = 0; agent < agents; ++agent) {
            std::string line;
            for (;;) {
                const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(
                    deadline - std::chrono::steady_clock::now());
                pollfd fd { ready_pipes[agent], POLLIN, 0 };
                char c;
                if (left.count() <= 0 || poll(&fd, 1, static_cast<int>(left.count())) <= 0
                    || read(ready_pipes[agent], &c, 1) != 1)
                    throw std::runtime_error("the flight software of agent " + std::to_string(agent)
                        + " never said it was ready");
                if (c != '\n') {
                    line += c;
                } else if (line == "ready") {
                    break;
                } else {
                    line.clear();
                }
            }
        }
    } catch (...) {
        for (int fd : ready_pipes)
            close(fd);
        stop();
        throw;
    }
    for (int fd : ready_pipes)
        close(fd);
}

ProcessExchange::~ProcessExchange() { stop(); }

void ProcessExchange::stop()
{
    // As Elodin stops its process recipes: SIGTERM, up to 2 seconds to exit, then SIGKILL.
    for (pid_t pid : children_)
        kill(pid, SIGTERM);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    for (pid_t pid : children_) {
        while (waitpid(pid, nullptr, WNOHANG) == 0) {
            if (std::chrono::steady_clock::now() > deadline) {
                kill(pid, SIGKILL);
                waitpid(pid, nullptr, 0);
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }
    children_.clear();
    for (int fd : receivers_)
        close(fd);
    receivers_.clear();
    if (sender_ >= 0)
        close(sender_);
    sender_ = -1;
}

void ProcessExchange::run(const uint8_t* states, uint8_t* replies)
{
    for (size_t agent = 0; agent < agents_; ++agent) {
        const sockaddr_in address = localhost(static_cast<uint16_t>(port_base_ + 2 * agent));
        if (sendto(sender_, states + agent * protocol::STATE_SIZE, protocol::STATE_SIZE, 0,
                reinterpret_cast<const sockaddr*>(&address), sizeof address)
            != static_cast<ssize_t>(protocol::STATE_SIZE))
            fail("send to agent " + std::to_string(agent));
    }
    std::vector<pollfd> pending;
    for (int fd : receivers_)
        pending.push_back({ fd, POLLIN, 0 });
    std::vector<bool> answered(agents_, false);
    size_t left = agents_;
    const auto deadline = std::chrono::steady_clock::now() + timeout_;
    while (left > 0) {
        const auto wait = std::chrono::duration_cast<std::chrono::milliseconds>(
            deadline - std::chrono::steady_clock::now());
        if (wait.count() < 0 || poll(pending.data(), pending.size(), static_cast<int>(wait.count())) <= 0)
            throw std::runtime_error("flight software response timeout");
        for (size_t agent = 0; agent < agents_; ++agent) {
            if (answered[agent] || !(pending[agent].revents & POLLIN))
                continue;
            const ssize_t size = recv(receivers_[agent], replies + agent * protocol::COMMAND_SIZE,
                protocol::COMMAND_SIZE, 0);
            if (size != static_cast<ssize_t>(protocol::COMMAND_SIZE))
                throw std::runtime_error("invalid command packet size");
            answered[agent] = true;
            pending[agent].events = 0;
            --left;
        }
    }
}

std::unique_ptr<Exchange> make_exchange(const std::string& transport, const std::string& fsw_binary,
    size_t agents, uint16_t port_base, std::chrono::milliseconds timeout)
{
    if (transport == "in-process")
        return std::make_unique<InProcessExchange>(agents);
    if (transport == "processes")
        return std::make_unique<ProcessExchange>(fsw_binary, agents, port_base, timeout);
    throw std::runtime_error("--fsw takes in-process or processes, not " + transport);
}
