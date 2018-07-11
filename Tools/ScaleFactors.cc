#include "ScaleFactors.h"

extern "C" 
{
    static float python_sf_norm0b()
    {
        return ScaleFactors::sf_norm0b();
    }
    static float python_sf_norm0b_err()
    {
        return ScaleFactors::sfunc_norm0b();
    }
}
