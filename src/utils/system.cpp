#include <cstdio>
#include <string.h>
#include <stdlib.h>

#include "trtengine/utils/logger.h"

// Function to get current memory usage (OS-dependent, simplified for demonstration)
// For accurate leak detection, a tool like Valgrind is recommended.
// This is a placeholder for observation.
long getCurrentRSS() {
#ifdef __linux__
    long rss = 0L;
    FILE* fp = NULL;
    if ( (fp = fopen( "/proc/self/status", "r" )) == NULL )
        return 0; // Can't open?
    char line[128];
    while ( fgets( line, 128, fp ) != NULL ) {
        if ( strncmp( line, "VmRSS:", 6 ) == 0 ) {
            rss = atol( line + 6 );
            break;
        }
    }
    fclose(fp);
    return rss; // in KB
#else
    // For Windows, macOS, or other OSes, you would implement similar logic
    LOG_ERROR("SYSTEM_UTILS", "getCurrentRSS is not implemented for this OS.");

    // Placeholder for other OSes
    return 0;
#endif
}