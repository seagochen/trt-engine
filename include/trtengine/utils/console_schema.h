/**
 * @file console_color.h
 * @brief ANSI escape codes for colored and formatted console output.
 *
 * This header defines macros for various text colors, styles, and backgrounds
 * using ANSI escape codes. These macros can be used to enhance console output
 * with colors and formatting for better readability and emphasis.
 *
 * Macro definitions:
 *
 * - CONSOLE_COLOR_RESET:        Reset all attributes to default.
 *
 * Regular Colors:
 * - CONSOLE_COLOR_BLACK:        Black foreground.
 * - CONSOLE_COLOR_RED:          Red foreground.
 * - CONSOLE_COLOR_GREEN:        Green foreground.
 * - CONSOLE_COLOR_YELLOW:       Yellow foreground.
 * - CONSOLE_COLOR_BLUE:         Blue foreground.
 * - CONSOLE_COLOR_PURPLE:       Purple (magenta) foreground.
 * - CONSOLE_COLOR_CYAN:         Cyan foreground.
 * - CONSOLE_COLOR_WHITE:        White foreground.
 * - CONSOLE_COLOR_GRAY:         Gray foreground.
 * - CONSOLE_COLOR_LIGHT_RED:    Light red foreground.
 * - CONSOLE_COLOR_LIGHT_GREEN:  Light green foreground.
 * - CONSOLE_COLOR_LIGHT_YELLOW: Light yellow foreground.
 * - CONSOLE_COLOR_LIGHT_BLUE:   Light blue foreground.
 * - CONSOLE_COLOR_LIGHT_PURPLE: Light purple (magenta) foreground.
 * - CONSOLE_COLOR_LIGHT_CYAN:   Light cyan foreground.
 * - CONSOLE_COLOR_LIGHT_GRAY:   Light gray foreground.
 *
 * Bold Colors:
 * - CONSOLE_COLOR_BBLACK:       Bold black foreground.
 * - CONSOLE_COLOR_BRED:         Bold red foreground.
 * - CONSOLE_COLOR_BGREEN:       Bold green foreground.
 * - CONSOLE_COLOR_BYELLOW:      Bold yellow foreground.
 * - CONSOLE_COLOR_BBLUE:        Bold blue foreground.
 * - CONSOLE_COLOR_BPURPLE:      Bold purple (magenta) foreground.
 * - CONSOLE_COLOR_BCYAN:        Bold cyan foreground.
 * - CONSOLE_COLOR_BWHITE:       Bold white foreground.
 *
 * Underline Colors:
 * - CONSOLE_COLOR_ULBLACK:      Underlined black foreground.
 * - CONSOLE_COLOR_ULRED:        Underlined red foreground.
 * - CONSOLE_COLOR_ULGREEN:      Underlined green foreground.
 * - CONSOLE_COLOR_ULYELLOW:     Underlined yellow foreground.
 * - CONSOLE_COLOR_ULBLUE:       Underlined blue foreground.
 * - CONSOLE_COLOR_ULPURPLE:     Underlined purple (magenta) foreground.
 * - CONSOLE_COLOR_ULCYAN:       Underlined cyan foreground.
 * - CONSOLE_COLOR_ULWHITE:      Underlined white foreground.
 *
 * Background Colors:
 * - CONSOLE_COLOR_ON_BLACK:     Black background.
 * - CONSOLE_COLOR_ON_RED:       Red background.
 * - CONSOLE_COLOR_ON_GREEN:     Green background.
 * - CONSOLE_COLOR_ON_YELLOW:    Yellow background.
 * - CONSOLE_COLOR_ON_BLUE:      Blue background.
 * - CONSOLE_COLOR_ON_PURPLE:    Purple (magenta) background.
 * - CONSOLE_COLOR_ON_CYAN:      Cyan background.
 * - CONSOLE_COLOR_ON_WHITE:     White background.
 *
 * High Intensity Colors:
 * - CONSOLE_COLOR_IBLACK:       High intensity black foreground.
 * - CONSOLE_COLOR_IRED:         High intensity red foreground.
 * - CONSOLE_COLOR_IGREEN:       High intensity green foreground.
 * - CONSOLE_COLOR_IYELLOW:      High intensity yellow foreground.
 * - CONSOLE_COLOR_IBLUE:        High intensity blue foreground.
 * - CONSOLE_COLOR_IPURPLE:      High intensity purple (magenta) foreground.
 * - CONSOLE_COLOR_ICYAN:        High intensity cyan foreground.
 * - CONSOLE_COLOR_IWHITE:       High intensity white foreground.
 *
 * Bold High Intensity Colors:
 * - CONSOLE_COLOR_BIBLACK:      Bold high intensity black foreground.
 * - CONSOLE_COLOR_BIRED:        Bold high intensity red foreground.
 * - CONSOLE_COLOR_BIGREEN:      Bold high intensity green foreground.
 * - CONSOLE_COLOR_BIYELLOW:     Bold high intensity yellow foreground.
 * - CONSOLE_COLOR_BIBLUE:       Bold high intensity blue foreground.
 * - CONSOLE_COLOR_BIPURPLE:     Bold high intensity purple (magenta) foreground.
 * - CONSOLE_COLOR_BICYAN:       Bold high intensity cyan foreground.
 * - CONSOLE_COLOR_BIWHITE:      Bold high intensity white foreground.
 *
 * High Intensity Backgrounds:
 * - CONSOLE_COLOR_ON_IBLACK:    High intensity black background.
 * - CONSOLE_COLOR_ON_IRED:      High intensity red background.
 * - CONSOLE_COLOR_ON_IGREEN:    High intensity green background.
 * - CONSOLE_COLOR_ON_IYELLOW:   High intensity yellow background.
 * - CONSOLE_COLOR_ON_IBLUE:     High intensity blue background.
 * - CONSOLE_COLOR_ON_IPURPLE:   High intensity purple (magenta) background.
 * - CONSOLE_COLOR_ON_ICYAN:     High intensity cyan background.
 * - CONSOLE_COLOR_ON_IWHITE:    High intensity white background.
 *
 * Usage example:
 *   std::cout << CONSOLE_COLOR_RED << "Error!" << CONSOLE_COLOR_RESET << std::endl;
 */

#ifndef COMBINEDPROJECT_CONSOLE_COLOR_H
#define COMBINEDPROJECT_CONSOLE_COLOR_H

// Reset
#define CONSOLE_COLOR_RESET        "\033[0m"

// Regular Colors
#define CONSOLE_COLOR_BLACK        "\033[0;30m"
#define CONSOLE_COLOR_RED          "\033[0;31m"
#define CONSOLE_COLOR_GREEN        "\033[0;32m"
#define CONSOLE_COLOR_YELLOW       "\033[0;33m"
#define CONSOLE_COLOR_BLUE         "\033[0;34m"
#define CONSOLE_COLOR_PURPLE       "\033[0;35m"
#define CONSOLE_COLOR_CYAN         "\033[0;36m"
#define CONSOLE_COLOR_WHITE        "\033[0;37m"
#define CONSOLE_COLOR_GRAY         "\033[0;90m"
#define CONSOLE_COLOR_LIGHT_RED    "\033[0;91m"
#define CONSOLE_COLOR_LIGHT_GREEN  "\033[0;92m"
#define CONSOLE_COLOR_LIGHT_YELLOW "\033[0;93m"
#define CONSOLE_COLOR_LIGHT_BLUE   "\033[0;94m"
#define CONSOLE_COLOR_LIGHT_PURPLE "\033[0;95m"
#define CONSOLE_COLOR_LIGHT_CYAN   "\033[0;96m"
#define CONSOLE_COLOR_LIGHT_GRAY   "\033[0;97m"

// Bold
#define CONSOLE_COLOR_BBLACK       "\033[1;30m"
#define CONSOLE_COLOR_BRED         "\033[1;31m"
#define CONSOLE_COLOR_BGREEN       "\033[1;32m"
#define CONSOLE_COLOR_BYELLOW      "\033[1;33m"
#define CONSOLE_COLOR_BBLUE        "\033[1;34m"
#define CONSOLE_COLOR_BPURPLE      "\033[1;35m"
#define CONSOLE_COLOR_BCYAN        "\033[1;36m"
#define CONSOLE_COLOR_BWHITE       "\033[1;37m"

// Underline
#define CONSOLE_COLOR_ULBLACK      "\033[4;30m"
#define CONSOLE_COLOR_ULRED        "\033[4;31m"
#define CONSOLE_COLOR_ULGREEN      "\033[4;32m"
#define CONSOLE_COLOR_ULYELLOW     "\033[4;33m"
#define CONSOLE_COLOR_ULBLUE       "\033[4;34m"
#define CONSOLE_COLOR_ULPURPLE     "\033[4;35m"
#define CONSOLE_COLOR_ULCYAN       "\033[4;36m"
#define CONSOLE_COLOR_ULWHITE      "\033[4;37m"

// Background
#define CONSOLE_COLOR_ON_BLACK     "\033[40m"
#define CONSOLE_COLOR_ON_RED       "\033[41m"
#define CONSOLE_COLOR_ON_GREEN     "\033[42m"
#define CONSOLE_COLOR_ON_YELLOW    "\033[43m"
#define CONSOLE_COLOR_ON_BLUE      "\033[44m"
#define CONSOLE_COLOR_ON_PURPLE    "\033[45m"
#define CONSOLE_COLOR_ON_CYAN      "\033[46m"
#define CONSOLE_COLOR_ON_WHITE     "\033[47m"

// High Intensity
#define CONSOLE_COLOR_IBLACK       "\033[0;90m"
#define CONSOLE_COLOR_IRED         "\033[0;91m"
#define CONSOLE_COLOR_IGREEN       "\033[0;92m"
#define CONSOLE_COLOR_IYELLOW      "\033[0;93m"
#define CONSOLE_COLOR_IBLUE        "\033[0;94m"
#define CONSOLE_COLOR_IPURPLE      "\033[0;95m"
#define CONSOLE_COLOR_ICYAN        "\033[0;96m"
#define CONSOLE_COLOR_IWHITE       "\033[0;97m"

// Bold High Intensity
#define CONSOLE_COLOR_BIBLACK      "\033[1;90m"
#define CONSOLE_COLOR_BIRED        "\033[1;91m"
#define CONSOLE_COLOR_BIGREEN      "\033[1;92m"
#define CONSOLE_COLOR_BIYELLOW     "\033[1;93m"
#define CONSOLE_COLOR_BIBLUE       "\033[1;94m"
#define CONSOLE_COLOR_BIPURPLE     "\033[1;95m"
#define CONSOLE_COLOR_BICYAN       "\033[1;96m"
#define CONSOLE_COLOR_BIWHITE      "\033[1;97m"

// High Intensity backgrounds
#define CONSOLE_COLOR_ON_IBLACK    "\033[0;100m"
#define CONSOLE_COLOR_ON_IRED      "\033[0;101m"
#define CONSOLE_COLOR_ON_IGREEN    "\033[0;102m"
#define CONSOLE_COLOR_ON_IYELLOW   "\033[0;103m"
#define CONSOLE_COLOR_ON_IBLUE     "\033[0;104m"
#define CONSOLE_COLOR_ON_IPURPLE   "\033[0;105m"
#define CONSOLE_COLOR_ON_ICYAN     "\033[0;106m"
#define CONSOLE_COLOR_ON_IWHITE    "\033[0;107m"

#endif // COMBINEDPROJECT_CONSOLE_COLOR_H
