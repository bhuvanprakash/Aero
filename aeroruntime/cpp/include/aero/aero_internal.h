/**
 * Internal API for AERO C library (not for public use).
 * Allows other .cpp files in the same library to set the last error.
 */
#ifndef AERO_AERO_INTERNAL_H
#define AERO_AERO_INTERNAL_H

#ifdef __cplusplus
extern "C" {
#endif

/** Set the last error message (used by aero_error_string()). */
void aero_internal_set_error(const char* msg);

#ifdef __cplusplus
}
#endif

#endif
