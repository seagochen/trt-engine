//
// Created by orlando on 9/25/24.
//

#ifndef SYS_SIGNAL_HPP
#define SYS_SIGNAL_HPP

#include <csignal>

volatile sig_atomic_t signal_received = 0;

void signal_handler(int signal) {

    switch (signal) {
        case SIGINT:
            std::cout << "Ctrl+C signal received" << std::endl;
            break;
        case SIGTSTP:
            std::cout << "Ctrl+Z signal received" << std::endl;
            break;
        case SIGQUIT:
            std::cout << "Ctrl+\\ signal received" << std::endl;
            break;
        case SIGTERM:
            std::cout << "Ctrl+D signal received" << std::endl;
            break;
        case SIGSTOP:
            std::cout << "Ctrl+Z signal received" << std::endl;
            break;
        default:
            std::cout << "Unknown signal received" << std::endl;
            break;
    }

    // Ctrl+C: SIGINT
    // Ctrl+Z: SIGTSTP
    // Ctrl+\: SIGQUIT
    // Ctrl+D: SIGTERM
    // Ctrl+Z: SIGSTOP
    signal_received = signal;
}

void register_sigint() {
    signal(SIGINT, signal_handler);
}

void register_sigtstp() {
    signal(SIGTSTP, signal_handler);
}

void register_sigquit() {
    signal(SIGQUIT, signal_handler);
}

void register_sigterm() {
    signal(SIGTERM, signal_handler);
}

void register_sigstop() {
    signal(SIGSTOP, signal_handler);
}

#endif //SYS_SIGNAL_HPP
