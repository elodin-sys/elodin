#!/usr/bin/env python3
"""Patch tegraflash_impl_t234.py to strip NVMe from MB2 storage info.

MB2 initializes every device listed in its BCT storage info. On the Aleph
carrier PCIe cannot come up that early: MB2 dies with an SError in
tegrabl_pcie_soc_init and the board falls back to recovery. Feed every
--updatestorageinfo call an nvme-stripped copy of the layout; GPT and
flash.idx generation keep using the full layout, and UEFI handles NVMe later.
"""

import sys
from pathlib import Path

HELPER = """
    def aleph_storage_pt(self):
        import re
        out_xml = 'aleph_storage_nonvme.xml'
        out_bin = os.path.splitext(out_xml)[0] + '.bin'
        if not os.path.exists(out_bin):
            with open(values['--cfg']) as f:
                xml_txt = f.read()
            xml_txt, count = re.subn(r'<device[^>]*type="nvme"[\\s\\S]*?</device>', '', xml_txt)
            if count != 1:
                raise RuntimeError(f'expected one nvme device, found {count}')
            with open(out_xml, 'w') as f:
                f.write(xml_txt)
            command = self.exec_file('tegraparser')
            command.extend(['--pt', out_xml])
            run_command(command)
        return out_bin

"""

ANCHOR = "    def tegraflash_fill_mb1_storage_info(self):"
OLD_ARG = "'--updatestorageinfo', self.tegraparser_values['--pt']"
NEW_ARG = "'--updatestorageinfo', self.aleph_storage_pt()"
EXPECTED_CALLS = 5


def main() -> None:
    path = Path(sys.argv[1])
    text = path.read_text()

    count = text.count(OLD_ARG)
    if count != EXPECTED_CALLS:
        raise SystemExit(f"expected {EXPECTED_CALLS} updatestorageinfo calls, found {count}")
    text = text.replace(OLD_ARG, NEW_ARG)

    if ANCHOR not in text:
        raise SystemExit("helper anchor not found")
    text = text.replace(ANCHOR, HELPER + ANCHOR, 1)

    path.write_text(text)
    print(f"Patched {path}: {count} storage-info call(s) now nvme-stripped")


if __name__ == "__main__":
    main()
