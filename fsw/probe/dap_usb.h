#ifndef DAP_USB_H
#define DAP_USB_H

#include "usbd.h"

static inline bool dap_usb_session_invalidated(Usbd_State previous_state,
                                                U32 previous_lifecycle,
                                                Usbd_State state,
                                                U32 lifecycle,
                                                bool endpoints_missing) {
    if (endpoints_missing || lifecycle != previous_lifecycle) return true;

    bool suspend_resume =
        (previous_state == UsbdState_Configured && state == UsbdState_Suspended) ||
        (previous_state == UsbdState_Suspended && state == UsbdState_Configured);
    return state != previous_state && !suspend_resume;
}

#endif
