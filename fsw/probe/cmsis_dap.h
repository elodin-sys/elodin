#ifndef CMSIS_DAP_H
#define CMSIS_DAP_H

#include "core.h"
#include "hal.h"
#include "swd.h"

enum {
    CmsisDapPacketSize = 64u,
    CmsisDapPacketCount = 4u,

    CmsisDapCommand_Info = 0x00u,
    CmsisDapCommand_HostStatus = 0x01u,
    CmsisDapCommand_Connect = 0x02u,
    CmsisDapCommand_Disconnect = 0x03u,
    CmsisDapCommand_TransferConfigure = 0x04u,
    CmsisDapCommand_Transfer = 0x05u,
    CmsisDapCommand_TransferBlock = 0x06u,
    CmsisDapCommand_TransferAbort = 0x07u,
    CmsisDapCommand_WriteAbort = 0x08u,
    CmsisDapCommand_Delay = 0x09u,
    CmsisDapCommand_ResetTarget = 0x0Au,
    CmsisDapCommand_SwjPins = 0x10u,
    CmsisDapCommand_SwjClock = 0x11u,
    CmsisDapCommand_SwjSequence = 0x12u,
    CmsisDapCommand_SwdConfigure = 0x13u,
    CmsisDapCommand_SwdSequence = 0x1Du,
    CmsisDapCommand_Vendor0 = 0x80u,
    CmsisDapCommand_Invalid = 0xFFu,

    CmsisDapInfo_Vendor = 0x01u,
    CmsisDapInfo_Product = 0x02u,
    CmsisDapInfo_Serial = 0x03u,
    CmsisDapInfo_ProtocolVersion = 0x04u,
    CmsisDapInfo_ProductFirmwareVersion = 0x09u,
    CmsisDapInfo_Capabilities = 0xF0u,
    CmsisDapInfo_PacketCount = 0xFEu,
    CmsisDapInfo_PacketSize = 0xFFu,

    CmsisDapPort_Autodetect = 0u,
    CmsisDapPort_Disabled = 0u,
    CmsisDapPort_Swd = 1u,
    CmsisDapPort_Jtag = 2u,

    CmsisDapStatus_Ok = 0u,
    CmsisDapStatus_Error = 0xFFu,

    CmsisDapTransfer_ApnDp = 1u << 0,
    CmsisDapTransfer_Read = 1u << 1,
    CmsisDapTransfer_MatchValue = 1u << 4,
    CmsisDapTransfer_MatchMask = 1u << 5,
    CmsisDapTransfer_Timestamp = 1u << 7,
    CmsisDapTransfer_Mismatch = 1u << 4,

    CmsisDapDp_Rdbuff = 0x0Cu | CmsisDapTransfer_Read,
};

typedef bool (*CmsisDap_AbortPoll)(void *context);

typedef struct {
    Swd swd;
    CmsisDap_AbortPoll abort_poll;
    void *abort_context;
    U32 command_count;
    U32 transfer_command_count;
    U32 malformed_count;
    U32 unsupported_count;
    U32 match_mask;
    U16 retry_count;
    U16 match_retry;
    U8 debug_port;
    bool debugger_connected;
    bool target_running;
    volatile bool transfer_abort;
    char const *firmware_version;
    char serial[17];
} CmsisDap;

static U8 cmsis_dap__string(U8 *out, U16 capacity, char const *text) {
    if (!out || capacity == 0u || !text) return 0u;

    U16 length = 0u;
    while (text[length] && (U16)(length + 1u) < capacity && length < 254u) {
        out[length] = (U8)text[length];
        length++;
    }
    if (length >= capacity) return 0u;
    out[length++] = 0u;
    return (U8)length;
}

static bool cmsis_dap__abort_requested(CmsisDap *dap) {
    if (dap->transfer_abort) return true;
    if (dap->abort_poll && dap->abort_poll(dap->abort_context)) {
        dap->transfer_abort = true;
    }
    return dap->transfer_abort;
}

static bool cmsis_dap__wait_us(CmsisDap *dap, U32 delay_us) {
    while (delay_us != 0u) {
        if (cmsis_dap__abort_requested(dap)) return false;
        U32 slice_us = Min(delay_us, 100u);
        hal_delay_us(slice_us);
        delay_us -= slice_us;
    }
    return !cmsis_dap__abort_requested(dap);
}

static U8 cmsis_dap__transfer_retry(CmsisDap *dap, U8 request, U32 *data) {
    U16 retry = dap->retry_count;
    U8 response;

    do {
        response = swd_transfer(&dap->swd, request, data);
        if (response != SwdAck_Wait || cmsis_dap__abort_requested(dap)) break;
    } while (retry-- != 0u);

    return response;
}

static U16 cmsis_dap__malformed(CmsisDap *dap, U8 command, U8 *response, U16 capacity) {
    dap->malformed_count++;
    if (!response || capacity < 2u) return 0u;
    response[0] = command;
    response[1] = CmsisDapStatus_Error;
    return 2u;
}

static U16 cmsis_dap__info(CmsisDap *dap, U8 const *request, U16 request_size,
                           U8 *response, U16 response_capacity) {
    if (request_size < 2u || response_capacity < 2u) {
        return cmsis_dap__malformed(dap, CmsisDapCommand_Info, response, response_capacity);
    }

    response[0] = CmsisDapCommand_Info;
    U8 length = 0u;
    switch (request[1]) {
    case CmsisDapInfo_Vendor:
        length = cmsis_dap__string(response + 2, (U16)(response_capacity - 2u), "Elodin");
        break;
    case CmsisDapInfo_Product:
        length = cmsis_dap__string(response + 2, (U16)(response_capacity - 2u), "Aleph FC CMSIS-DAP");
        break;
    case CmsisDapInfo_Serial:
        length = cmsis_dap__string(response + 2, (U16)(response_capacity - 2u), dap->serial);
        break;
    case CmsisDapInfo_ProtocolVersion:
        length = cmsis_dap__string(response + 2, (U16)(response_capacity - 2u), "2.1.2");
        break;
    case CmsisDapInfo_ProductFirmwareVersion:
        length = cmsis_dap__string(response + 2, (U16)(response_capacity - 2u), dap->firmware_version);
        break;
    case CmsisDapInfo_Capabilities:
        if (response_capacity >= 4u) {
            response[2] = 1u << 0; // SWD only; no atomic, JTAG, SWO, or UART claims.
            response[3] = 0u;
            length = 2u;
        }
        break;
    case CmsisDapInfo_PacketCount:
        if (response_capacity >= 3u) {
            response[2] = CmsisDapPacketCount;
            length = 1u;
        }
        break;
    case CmsisDapInfo_PacketSize:
        if (response_capacity >= 4u) {
            write_le16(response + 2, CmsisDapPacketSize);
            length = 2u;
        }
        break;
    default:
        break;
    }

    response[1] = length;
    return (U16)(2u + length);
}

static U16 cmsis_dap__host_status(CmsisDap *dap, U8 const *request, U16 request_size,
                                  U8 *response, U16 response_capacity) {
    if (request_size < 3u || response_capacity < 2u) {
        return cmsis_dap__malformed(dap, CmsisDapCommand_HostStatus, response, response_capacity);
    }

    U8 status = CmsisDapStatus_Ok;
    if (request[1] == 0u) dap->debugger_connected = (request[2] & 1u) != 0u;
    else if (request[1] == 1u) dap->target_running = (request[2] & 1u) != 0u;
    else status = CmsisDapStatus_Error;

    response[0] = CmsisDapCommand_HostStatus;
    response[1] = status;
    return 2u;
}

static U16 cmsis_dap__connect(CmsisDap *dap, U8 const *request, U16 request_size,
                              U8 *response, U16 response_capacity) {
    if (request_size < 2u || response_capacity < 2u) {
        return cmsis_dap__malformed(dap, CmsisDapCommand_Connect, response, response_capacity);
    }

    U8 port = request[1];
    if (port == CmsisDapPort_Autodetect) port = CmsisDapPort_Swd;
    if (port == CmsisDapPort_Swd) {
        swd_connect(&dap->swd);
        dap->debug_port = CmsisDapPort_Swd;
    } else {
        port = CmsisDapPort_Disabled;
    }

    response[0] = CmsisDapCommand_Connect;
    response[1] = port;
    return 2u;
}

static U16 cmsis_dap__disconnect(CmsisDap *dap, U8 *response, U16 response_capacity) {
    if (response_capacity < 2u) return 0u;
    swd_disconnect(&dap->swd);
    dap->debug_port = CmsisDapPort_Disabled;
    response[0] = CmsisDapCommand_Disconnect;
    response[1] = CmsisDapStatus_Ok;
    return 2u;
}

static U16 cmsis_dap__transfer_configure(CmsisDap *dap, U8 const *request, U16 request_size,
                                         U8 *response, U16 response_capacity) {
    if (request_size < 6u || response_capacity < 2u) {
        return cmsis_dap__malformed(dap, CmsisDapCommand_TransferConfigure, response, response_capacity);
    }

    dap->swd.idle_cycles = request[1];
    dap->retry_count = read_le16(request + 2);
    dap->match_retry = read_le16(request + 4);
    response[0] = CmsisDapCommand_TransferConfigure;
    response[1] = CmsisDapStatus_Ok;
    return 2u;
}

static bool cmsis_dap__transfer_shape(U8 const *request, U16 request_size,
                                      U16 response_capacity, U16 *request_used,
                                      U16 *response_used) {
    if (request_size < 3u) return false;

    U16 in = 3u;
    U16 out = 3u;
    U8 count = request[2];
    for (U16 i = 0u; i < count; i++) {
        if (in >= request_size) return false;
        U8 transfer = request[in++];
        if ((transfer & CmsisDapTransfer_Timestamp) != 0u) return false;

        bool read = (transfer & CmsisDapTransfer_Read) != 0u;
        bool has_data = !read || (transfer & CmsisDapTransfer_MatchValue) != 0u;
        if (has_data) {
            if ((U32)in + 4u > request_size) return false;
            in = (U16)(in + 4u);
        }
        if (read && (transfer & CmsisDapTransfer_MatchValue) == 0u) {
            if ((U32)out + 4u > response_capacity) return false;
            out = (U16)(out + 4u);
        }
    }

    *request_used = in;
    *response_used = out;
    return true;
}

static U16 cmsis_dap__transfer(CmsisDap *dap, U8 const *request, U16 request_size,
                               U8 *response, U16 response_capacity) {
    U16 request_used = 0u;
    U16 response_limit = 0u;
    if (!response || response_capacity < 3u ||
        !cmsis_dap__transfer_shape(request, request_size, response_capacity,
                                   &request_used, &response_limit)) {
        return cmsis_dap__malformed(dap, CmsisDapCommand_Transfer, response, response_capacity);
    }
    (void)request_used;
    (void)response_limit;

    response[0] = CmsisDapCommand_Transfer;
    response[1] = 0u;
    response[2] = 0u;
    if (dap->debug_port != CmsisDapPort_Swd) return 3u;

    dap->transfer_command_count++;

    U16 in = 3u;
    U16 out = 3u;
    U8 count = request[2];
    U8 completed = 0u;
    U8 status = 0u;
    bool posted_read = false;
    bool check_write = false;

    for (U16 index = 0u; index < count; index++) {
        U8 transfer = request[in++];
        bool read = (transfer & CmsisDapTransfer_Read) != 0u;
        bool ap = (transfer & CmsisDapTransfer_ApnDp) != 0u;

        if (read) {
            if (posted_read) {
                U32 data = 0u;
                if (ap && (transfer & CmsisDapTransfer_MatchValue) == 0u) {
                    status = cmsis_dap__transfer_retry(dap, transfer, &data);
                } else {
                    status = cmsis_dap__transfer_retry(dap, CmsisDapDp_Rdbuff, &data);
                    posted_read = false;
                }
                if (status != SwdAck_Ok) break;
                write_le32(response + out, data);
                out = (U16)(out + 4u);
            }

            if ((transfer & CmsisDapTransfer_MatchValue) != 0u) {
                U32 match_value = read_le32(request + in);
                in = (U16)(in + 4u);
                U32 data = 0u;

                if (ap) {
                    status = cmsis_dap__transfer_retry(dap, transfer, 0);
                    if (status != SwdAck_Ok) break;
                }

                U16 match_retry = dap->match_retry;
                do {
                    status = cmsis_dap__transfer_retry(dap, transfer, &data);
                    if (status != SwdAck_Ok || cmsis_dap__abort_requested(dap)) break;
                    if ((data & dap->match_mask) == match_value) break;
                } while (match_retry-- != 0u);

                if (status == SwdAck_Ok && (data & dap->match_mask) != match_value) {
                    status |= CmsisDapTransfer_Mismatch;
                }
                if (status != SwdAck_Ok) break;
            } else if (ap) {
                if (!posted_read) {
                    status = cmsis_dap__transfer_retry(dap, transfer, 0);
                    if (status != SwdAck_Ok) break;
                    posted_read = true;
                }
            } else {
                U32 data = 0u;
                status = cmsis_dap__transfer_retry(dap, transfer, &data);
                if (status != SwdAck_Ok) break;
                write_le32(response + out, data);
                out = (U16)(out + 4u);
            }
            check_write = false;
        } else {
            if (posted_read) {
                U32 data = 0u;
                status = cmsis_dap__transfer_retry(dap, CmsisDapDp_Rdbuff, &data);
                if (status != SwdAck_Ok) break;
                write_le32(response + out, data);
                out = (U16)(out + 4u);
                posted_read = false;
            }

            U32 data = read_le32(request + in);
            in = (U16)(in + 4u);
            if ((transfer & CmsisDapTransfer_MatchMask) != 0u) {
                dap->match_mask = data;
                status = SwdAck_Ok;
            } else {
                status = cmsis_dap__transfer_retry(dap, transfer, &data);
                if (status != SwdAck_Ok) break;
                check_write = true;
            }
        }

        completed++;
        if (cmsis_dap__abort_requested(dap)) break;
    }

    if (status == SwdAck_Ok) {
        if (posted_read) {
            U32 data = 0u;
            status = cmsis_dap__transfer_retry(dap, CmsisDapDp_Rdbuff, &data);
            if (status == SwdAck_Ok) {
                write_le32(response + out, data);
                out = (U16)(out + 4u);
            }
        } else if (check_write) {
            status = cmsis_dap__transfer_retry(dap, CmsisDapDp_Rdbuff, 0);
        }
    }

    response[1] = completed;
    response[2] = status;
    return out;
}

static U16 cmsis_dap__transfer_block(CmsisDap *dap, U8 const *request, U16 request_size,
                                     U8 *response, U16 response_capacity) {
    if (request_size < 5u || response_capacity < 4u) {
        return cmsis_dap__malformed(dap, CmsisDapCommand_TransferBlock, response, response_capacity);
    }

    U16 count = read_le16(request + 2);
    U8 transfer = request[4];
    bool read = (transfer & CmsisDapTransfer_Read) != 0u;
    U32 needed_request = read ? 5u : 5u + ((U32)count * 4u);
    U32 needed_response = read ? 4u + ((U32)count * 4u) : 4u;
    if ((transfer & 0xF0u) != 0u || needed_request > request_size ||
        needed_response > response_capacity) {
        return cmsis_dap__malformed(dap, CmsisDapCommand_TransferBlock, response, response_capacity);
    }

    response[0] = CmsisDapCommand_TransferBlock;
    write_le16(response + 1, 0u);
    response[3] = 0u;
    if (dap->debug_port != CmsisDapPort_Swd || count == 0u) return 4u;

    dap->transfer_command_count++;

    U16 completed = 0u;
    U16 in = 5u;
    U16 out = 4u;
    U8 status = SwdAck_Ok;
    bool ap = (transfer & CmsisDapTransfer_ApnDp) != 0u;

    if (read) {
        if (ap) {
            status = cmsis_dap__transfer_retry(dap, transfer, 0);
        }
        while (status == SwdAck_Ok && completed < count && !cmsis_dap__abort_requested(dap)) {
            U8 active_request = transfer;
            if (ap && (U16)(completed + 1u) == count) active_request = CmsisDapDp_Rdbuff;

            U32 data = 0u;
            status = cmsis_dap__transfer_retry(dap, active_request, &data);
            if (status != SwdAck_Ok) break;
            write_le32(response + out, data);
            out = (U16)(out + 4u);
            completed++;
        }
    } else {
        while (completed < count && !cmsis_dap__abort_requested(dap)) {
            U32 data = read_le32(request + in);
            in = (U16)(in + 4u);
            status = cmsis_dap__transfer_retry(dap, transfer, &data);
            if (status != SwdAck_Ok) break;
            completed++;
        }
        if (status == SwdAck_Ok && !dap->transfer_abort) {
            status = cmsis_dap__transfer_retry(dap, CmsisDapDp_Rdbuff, 0);
        }
    }

    write_le16(response + 1, completed);
    response[3] = status;
    return out;
}

static U16 cmsis_dap__write_abort(CmsisDap *dap, U8 const *request, U16 request_size,
                                  U8 *response, U16 response_capacity) {
    if (request_size < 6u || response_capacity < 2u) {
        return cmsis_dap__malformed(dap, CmsisDapCommand_WriteAbort, response, response_capacity);
    }

    U8 status = CmsisDapStatus_Error;
    if (dap->debug_port == CmsisDapPort_Swd) {
        U32 data = read_le32(request + 2);
        status = swd_transfer(&dap->swd, 0u, &data) == SwdAck_Ok ? CmsisDapStatus_Ok : CmsisDapStatus_Error;
    }
    response[0] = CmsisDapCommand_WriteAbort;
    response[1] = status;
    return 2u;
}

static U16 cmsis_dap__delay(CmsisDap *dap, U8 const *request, U16 request_size,
                            U8 *response, U16 response_capacity) {
    if (request_size < 3u || response_capacity < 2u) {
        return cmsis_dap__malformed(dap, CmsisDapCommand_Delay, response, response_capacity);
    }
    bool completed = cmsis_dap__wait_us(dap, read_le16(request + 1));
    response[0] = CmsisDapCommand_Delay;
    response[1] = completed ? CmsisDapStatus_Ok : CmsisDapStatus_Error;
    return 2u;
}

static U16 cmsis_dap__reset_target(CmsisDap *dap, U8 *response, U16 response_capacity) {
    if (response_capacity < 3u) return 0u;
    swd_nreset_write(&dap->swd, false);
    bool completed = cmsis_dap__wait_us(dap, 10000u);
    swd_nreset_write(&dap->swd, true);
    response[0] = CmsisDapCommand_ResetTarget;
    response[1] = completed ? CmsisDapStatus_Ok : CmsisDapStatus_Error;
    response[2] = 1u;
    return 3u;
}

static U16 cmsis_dap__swj_pins(CmsisDap *dap, U8 const *request, U16 request_size,
                               U8 *response, U16 response_capacity) {
    if (request_size < 7u || response_capacity < 2u) {
        return cmsis_dap__malformed(dap, CmsisDapCommand_SwjPins, response, response_capacity);
    }

    U8 value = request[1];
    U8 select = request[2];
    U32 wait_us = read_le32(request + 3);
    if (wait_us > 3000000u) wait_us = 3000000u;

    swd_pins_write(&dap->swd, value, select);
    U8 supported = (1u << 0) | (1u << 1) | (1u << 7);
    U8 compare = select & supported;
    if (wait_us != 0u && compare != 0u) {
        U32 start = hal_now_us();
        while ((swd_pins_read(&dap->swd) & compare) != (value & compare)) {
            if (hal_time_expired_us(start, wait_us)) break;
        }
    }

    response[0] = CmsisDapCommand_SwjPins;
    response[1] = swd_pins_read(&dap->swd);
    return 2u;
}

static U16 cmsis_dap__swj_clock(CmsisDap *dap, U8 const *request, U16 request_size,
                                U8 *response, U16 response_capacity) {
    if (request_size < 5u || response_capacity < 2u) {
        return cmsis_dap__malformed(dap, CmsisDapCommand_SwjClock, response, response_capacity);
    }

    response[0] = CmsisDapCommand_SwjClock;
    response[1] = swd_set_clock(&dap->swd, read_le32(request + 1)) ? CmsisDapStatus_Ok : CmsisDapStatus_Error;
    return 2u;
}

static U16 cmsis_dap__swj_sequence(CmsisDap *dap, U8 const *request, U16 request_size,
                                   U8 *response, U16 response_capacity) {
    if (request_size < 2u || response_capacity < 2u) {
        return cmsis_dap__malformed(dap, CmsisDapCommand_SwjSequence, response, response_capacity);
    }

    U16 count = request[1] ? request[1] : 256u;
    U16 bytes = (U16)((count + 7u) / 8u);
    if ((U32)bytes + 2u > request_size) {
        return cmsis_dap__malformed(dap, CmsisDapCommand_SwjSequence, response, response_capacity);
    }

    swd_sequence_out(&dap->swd, count, request + 2);
    response[0] = CmsisDapCommand_SwjSequence;
    response[1] = CmsisDapStatus_Ok;
    return 2u;
}

static U16 cmsis_dap__swd_configure(CmsisDap *dap, U8 const *request, U16 request_size,
                                    U8 *response, U16 response_capacity) {
    if (request_size < 2u || response_capacity < 2u) {
        return cmsis_dap__malformed(dap, CmsisDapCommand_SwdConfigure, response, response_capacity);
    }

    dap->swd.turnaround = (U8)((request[1] & 3u) + 1u);
    dap->swd.data_phase = (request[1] & 4u) != 0u;
    response[0] = CmsisDapCommand_SwdConfigure;
    response[1] = CmsisDapStatus_Ok;
    return 2u;
}

static U16 cmsis_dap__swd_sequence(CmsisDap *dap, U8 const *request, U16 request_size,
                                   U8 *response, U16 response_capacity) {
    if (request_size < 2u || response_capacity < 2u) {
        return cmsis_dap__malformed(dap, CmsisDapCommand_SwdSequence, response, response_capacity);
    }

    U16 in = 2u;
    U16 out = 2u;
    U8 sequence_count = request[1];
    for (U16 sequence = 0u; sequence < sequence_count; sequence++) {
        if (in >= request_size) {
            return cmsis_dap__malformed(dap, CmsisDapCommand_SwdSequence, response, response_capacity);
        }
        U8 info = request[in++];
        U16 bits = (U16)(info & 0x3Fu);
        if (bits == 0u) bits = 64u;
        U16 bytes = (U16)((bits + 7u) / 8u);
        if ((info & 0x80u) != 0u) {
            if ((U32)out + bytes > response_capacity) {
                return cmsis_dap__malformed(dap, CmsisDapCommand_SwdSequence, response, response_capacity);
            }
            out = (U16)(out + bytes);
        } else {
            if ((U32)in + bytes > request_size) {
                return cmsis_dap__malformed(dap, CmsisDapCommand_SwdSequence, response, response_capacity);
            }
            in = (U16)(in + bytes);
        }
    }

    response[0] = CmsisDapCommand_SwdSequence;
    response[1] = CmsisDapStatus_Ok;
    in = 2u;
    out = 2u;
    for (U16 sequence = 0u; sequence < sequence_count; sequence++) {
        U8 info = request[in++];
        U16 bits = (U16)(info & 0x3Fu);
        if (bits == 0u) bits = 64u;
        U16 bytes = (U16)((bits + 7u) / 8u);

        if ((info & 0x80u) != 0u) {
            swd_sequence_in(&dap->swd, bits, response + out);
            out = (U16)(out + bytes);
        } else {
            swd_sequence_out(&dap->swd, bits, request + in);
            in = (U16)(in + bytes);
        }
    }
    swd_idle(&dap->swd);
    return out;
}

static bool cmsis_dap_init(CmsisDap *dap, Bsp_SwdPort const *port,
                           char const *serial, char const *firmware_version,
                           CmsisDap_AbortPoll abort_poll, void *abort_context) {
    if (!dap || !port || !serial || !firmware_version) return false;

    *dap = (CmsisDap){
        .abort_poll = abort_poll,
        .abort_context = abort_context,
        .firmware_version = firmware_version,
        .retry_count = 100u,
        .match_retry = 0u,
        .match_mask = 0u,
    };
    for (U16 i = 0u; i < 16u; i++) {
        if (!serial[i]) return false;
        dap->serial[i] = serial[i];
    }
    dap->serial[16] = 0;
    return swd_init(&dap->swd, port);
}

static void cmsis_dap_disconnect(CmsisDap *dap) {
    if (!dap) return;
    swd_disconnect(&dap->swd);
    dap->debug_port = CmsisDapPort_Disabled;
    dap->transfer_abort = true;
}

static void cmsis_dap_abort(CmsisDap *dap) {
    if (dap) dap->transfer_abort = true;
}

static U16 cmsis_dap_execute(CmsisDap *dap, U8 const *request, U16 request_size,
                             U8 *response, U16 response_capacity) {
    if (!dap || !request || request_size == 0u || !response || response_capacity == 0u) return 0u;
    dap->transfer_abort = false;
    dap->command_count++;

    switch (request[0]) {
    case CmsisDapCommand_Info:
        return cmsis_dap__info(dap, request, request_size, response, response_capacity);
    case CmsisDapCommand_HostStatus:
        return cmsis_dap__host_status(dap, request, request_size, response, response_capacity);
    case CmsisDapCommand_Connect:
        return cmsis_dap__connect(dap, request, request_size, response, response_capacity);
    case CmsisDapCommand_Disconnect:
        return cmsis_dap__disconnect(dap, response, response_capacity);
    case CmsisDapCommand_TransferConfigure:
        return cmsis_dap__transfer_configure(dap, request, request_size, response, response_capacity);
    case CmsisDapCommand_Transfer:
        return cmsis_dap__transfer(dap, request, request_size, response, response_capacity);
    case CmsisDapCommand_TransferBlock:
        return cmsis_dap__transfer_block(dap, request, request_size, response, response_capacity);
    case CmsisDapCommand_TransferAbort:
        cmsis_dap_abort(dap);
        return 0u;
    case CmsisDapCommand_WriteAbort:
        return cmsis_dap__write_abort(dap, request, request_size, response, response_capacity);
    case CmsisDapCommand_Delay:
        return cmsis_dap__delay(dap, request, request_size, response, response_capacity);
    case CmsisDapCommand_ResetTarget:
        return cmsis_dap__reset_target(dap, response, response_capacity);
    case CmsisDapCommand_SwjPins:
        return cmsis_dap__swj_pins(dap, request, request_size, response, response_capacity);
    case CmsisDapCommand_SwjClock:
        return cmsis_dap__swj_clock(dap, request, request_size, response, response_capacity);
    case CmsisDapCommand_SwjSequence:
        return cmsis_dap__swj_sequence(dap, request, request_size, response, response_capacity);
    case CmsisDapCommand_SwdConfigure:
        return cmsis_dap__swd_configure(dap, request, request_size, response, response_capacity);
    case CmsisDapCommand_SwdSequence:
        return cmsis_dap__swd_sequence(dap, request, request_size, response, response_capacity);
    default:
        dap->unsupported_count++;
        response[0] = CmsisDapCommand_Invalid;
        return 1u;
    }
}

#endif
