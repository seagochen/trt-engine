#ifndef COMBINEDPROJECT_SYSTEM_H
#define COMBINEDPROJECT_SYSTEM_H


/**
 * @brief Get the current Resident Set Size (RSS) of the process.
 * This function returns the amount of memory currently used by the process.
 * The RSS is the portion of memory occupied by a process that is held in RAM.
 * It includes all memory that is currently allocated and in use by the process,
 * excluding memory that has been swapped out to disk.
 * @return The current RSS in bytes.
 */
long getCurrentRSS();

#endif // COMBINEDPROJECT_SYSTEM_H