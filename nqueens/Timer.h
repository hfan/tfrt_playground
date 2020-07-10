#pragma once

#include <stdint.h>
#include <time.h>

class Timer
{
public:
    void start_timer()
    { ::clock_gettime(CLOCK_REALTIME, &start); }

    void end_timer()
    {
        ::clock_gettime(CLOCK_REALTIME, &end);
        timed_interval = (int64_t)(1000000*(end.tv_sec - start.tv_sec) +
                                   (float)(end.tv_nsec - start.tv_nsec)/1000.0);
    }

    int64_t get_timed_interval()
    { return timed_interval; }

private:
    ::timespec start, end;
    int64_t timed_interval; // in microseconds
};
