//
// Created by orlando on 9/25/24.
//

#ifndef SYS_SIGNAL_HPP
#define SYS_SIGNAL_HPP

#include <csignal>

inline volatile sig_atomic_t signal_received = 0;

inline void signalHandler(int signal) {

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

inline sig_atomic_t getSigStatus() {
    return signal_received;
}

inline void registerSIGINT() {
    signal(SIGINT, signalHandler);
}

inline void registerSIGSTP() {
    signal(SIGTSTP, signalHandler);
}

inline void registerSIGQUIT() {
    signal(SIGQUIT, signalHandler);
}

inline void registerSIGTERM() {
    signal(SIGTERM, signalHandler);
}

inline void registerSIGSTOP() {
    signal(SIGSTOP, signalHandler);
}

#endif //SYS_SIGNAL_HPP
