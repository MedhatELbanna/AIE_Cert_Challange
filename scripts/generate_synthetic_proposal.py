"""
Generate a synthetic BMS Technical Submittal PDF for compliance testing.

The document is intentionally crafted with a mix of:
  - COMPLIANT items  (clearly met requirements)
  - PARTIAL items    (partially addressed, gaps present)
  - NON_COMPLIANT    (contradicts or fails spec requirement)
  - NOT_ADDRESSED    (silently omitted from proposal)

Run:
    python scripts/generate_synthetic_proposal.py
Output:
    docs/Synthetic-BMS-Proposal.pdf
"""

from __future__ import annotations

try:
    from fpdf import FPDF
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "fpdf2"])
    from fpdf import FPDF

from pathlib import Path

OUTPUT = Path(__file__).parent.parent / "docs" / "Synthetic-BMS-Proposal.pdf"


def _ascii(text: str) -> str:
    """Replace non-Latin-1 characters so fpdf core fonts don't choke."""
    return (
        text
        .replace("\u2014", "--")   # em dash
        .replace("\u2013", "-")    # en dash
        .replace("\u2022", "-")    # bullet
        .replace("\u2019", "'")    # right single quote
        .replace("\u2018", "'")    # left single quote
        .replace("\u201c", '"')    # left double quote
        .replace("\u201d", '"')    # right double quote
        .replace("\u2122", "(TM)") # trademark
        .replace("\u00b1", "+/-")  # plus-minus
    )

# ---------------------------------------------------------------------------
# Document content — structured as (section_title, body_paragraphs_list)
# Inline compliance notes mark intent for testers; they are NOT in the PDF.
# ---------------------------------------------------------------------------

COVER = {
    "company": "Nexus Controls & Automation, LLC",
    "project": "Riverside Medical Center — Building Management System",
    "submittal": "BMS-SUB-2025-047  |  Rev. 2",
    "date": "March 2025",
    "prepared_by": "J. Hartley, PE  —  Controls Engineer",
    "submitted_to": "Riverside Medical Center, Facilities Division",
}

SECTIONS = [
    # -----------------------------------------------------------------------
    # 1. EXECUTIVE SUMMARY
    # -----------------------------------------------------------------------
    (
        "1. Executive Summary",
        [
            "Nexus Controls & Automation, LLC (Nexus) is pleased to present this Technical "
            "Submittal for the Building Management System (BMS) at Riverside Medical Center. "
            "The proposed system is based on the Nexus APEX™ Direct Digital Control (DDC) "
            "platform, a fully open-protocol solution built on BACnet/IP (ANSI/ASHRAE Standard "
            "135) and compliant with ASHRAE Guideline 36-2021 High Performance Sequences of "
            "Operation.",

            "The system will monitor and control HVAC, domestic hot water, lighting, and "
            "electrical sub-metering throughout the facility. Integration with third-party "
            "systems — including the fire alarm panel and access control — will be provided "
            "through defined interfaces as described in Section 6.",

            "This submittal addresses the requirements of Specification Section 25 09 23 — "
            "Direct Digital Control (DDC) Systems for HVAC, and related Division 26 and "
            "Division 28 interfaces.",
        ],
    ),

    # -----------------------------------------------------------------------
    # 2. SYSTEM ARCHITECTURE
    # -----------------------------------------------------------------------
    (
        "2. System Architecture",
        [
            "2.1  Network Topology\n"
            "The BMS network follows a three-tier architecture:\n"
            "  • Tier 1 — Supervisory Layer: Nexus APEX Server running on a dedicated "
            "workstation (Windows Server 2022), connected to the facility IT backbone via "
            "a dedicated VLAN on 1 Gbps Ethernet.\n"
            "  • Tier 2 — Automation Layer: NX-4000 Building Automation Controllers (BACs) "
            "installed in each mechanical room, communicating via BACnet/IP over Cat6A.\n"
            "  • Tier 3 — Field Layer: NX-200 Application Specific Controllers (ASCs) and "
            "NX-110 Unitary Controllers on BACnet MS/TP at 76.8 kbps.",

            # COMPLIANT — BACnet/IP clearly stated
            "2.2  Communication Protocols\n"
            "All controllers are BTL-listed for BACnet/IP (B-BC profile) and BACnet MS/TP "
            "(B-ASC profile). The supervisory server exposes a BACnet Device Object (Device "
            "ID 1001) and supports the following BACnet services: ReadProperty, "
            "WriteProperty, SubscribeCOV, Who-Is/I-Am, and Time Synchronization. "
            "BACnet Alarm and Event services (AE) are supported per Annex N of ASHRAE 135.",

            # PARTIAL — Modbus mentioned but no specifics on chiller integration method
            "2.3  Third-Party Protocol Support\n"
            "The APEX Server includes a Modbus TCP/RTU gateway license. Modbus integration "
            "is available for equipment that does not support BACnet natively. Specific "
            "register maps for third-party equipment will be submitted separately during "
            "the submittal process as manufacturer data is confirmed.",

            "2.4  Redundancy\n"
            "The supervisory server is configured with RAID-1 mirrored storage. "
            "Field controllers operate autonomously (standalone mode) if network "
            "connectivity to the supervisory layer is lost; all control sequences "
            "continue uninterrupted. Failover time is less than 2 seconds.",
        ],
    ),

    # -----------------------------------------------------------------------
    # 3. HARDWARE
    # -----------------------------------------------------------------------
    (
        "3. Hardware Components",
        [
            "3.1  Supervisory Workstation — Nexus APEX Server\n"
            "  • CPU: Intel Xeon E-2300 series, 8-core\n"
            "  • RAM: 32 GB ECC DDR4\n"
            "  • Storage: 2 × 2 TB SSD (RAID-1)\n"
            "  • OS: Windows Server 2022 Standard\n"
            "  • UPS: APC Smart-UPS 1500VA, minimum 30-minute runtime\n"
            "  • Display: Dual 27-inch monitors (operator workstation)",

            "3.2  Building Automation Controllers — NX-4000 BAC\n"
            "  • Processor: 32-bit ARM Cortex-A7, 800 MHz\n"
            "  • Memory: 256 MB Flash, 128 MB RAM\n"
            "  • I/O: 8 UI, 4 AO, 6 DO (expandable via NX-IO modules)\n"
            "  • Communications: BACnet/IP (primary), BACnet MS/TP (field bus), RS-485\n"
            "  • Power: 24 VAC/VDC, Class 2\n"
            "  • UL Listed: UL 916 (Energy Management Equipment)",

            "3.3  Application Specific Controllers — NX-200 ASC\n"
            "  • Protocol: BACnet MS/TP (B-ASC)\n"
            "  • I/O: 4 UI, 2 AO, 4 DO\n"
            "  • Application: VAV terminal units, fan coil units, heat pump units\n"
            "  • UL Listed: UL 916",

            # NON_COMPLIANT — Emergency Power Off (EPO) integration not mentioned
            "3.4  Sensors and Actuators\n"
            "Temperature sensors: Nexus NX-TS10 (±0.2°C accuracy, NIST-traceable). "
            "CO2 sensors: Nexus NX-CO2-D, 0–2000 ppm range, NDIR technology, "
            "factory-calibrated. Pressure sensors: 0–5 in. w.c., ±0.5% FS. "
            "All actuators are spring-return type with manual override capability.",
        ],
    ),

    # -----------------------------------------------------------------------
    # 4. SOFTWARE & GRAPHICS
    # -----------------------------------------------------------------------
    (
        "4. Software and Operator Interface",
        [
            # COMPLIANT — Graphical UI with floor plans
            "4.1  Graphical User Interface\n"
            "The APEX Graphics™ web client provides HTML5-based operator access from "
            "any modern browser without plugins. Features include:\n"
            "  • Fully animated system schematics and floor plan graphics\n"
            "  • Drag-and-drop point scheduling\n"
            "  • Real-time trending and historical data display\n"
            "  • Role-based access control (RBAC) with configurable permission levels\n"
            "  • Mobile-responsive layout for tablet and smartphone access",

            # COMPLIANT — Trend logging
            "4.2  Data Logging and Trending\n"
            "All analog points are logged at configurable intervals (default: 5 minutes). "
            "Digital points are logged on change-of-value (COV). The system stores a "
            "minimum of 3 years of trend data on the local server. Data is archived to "
            "network-attached storage (NAS) automatically. Export formats: CSV, Excel, "
            "and BACnet Trend Log objects.",

            # COMPLIANT — Alarm management (basic)
            "4.3  Alarm Management\n"
            "The alarm subsystem complies with BACnet AE services. Alarms are classified "
            "into four priority tiers:\n"
            "  Priority 1 — Life Safety (immediate notification)\n"
            "  Priority 2 — Equipment Failure (5-minute acknowledgement required)\n"
            "  Priority 3 — Efficiency/Warning (15-minute acknowledgement)\n"
            "  Priority 4 — Informational (log only)\n"
            "Email and SMS notification is provided via the APEX AlertManager module. "
            "Alarm history is retained for a minimum of 3 years.",

            # PARTIAL — Alarm escalation mentioned but no time-based escalation schedule
            "  Note: Alarm escalation routing is configurable per customer requirements "
            "and will be defined during commissioning in coordination with the facilities "
            "team. A detailed escalation matrix will be provided as a separate deliverable.",

            "4.4  Scheduling\n"
            "The system supports weekly, holiday, and exception schedules. Up to 32 "
            "schedule objects per controller, each with up to 255 schedule entries. "
            "Daylight saving time is handled automatically.",
        ],
    ),

    # -----------------------------------------------------------------------
    # 5. HVAC CONTROL SEQUENCES
    # -----------------------------------------------------------------------
    (
        "5. HVAC Control Sequences",
        [
            # COMPLIANT — AHU control
            "5.1  Air Handling Units (AHUs)\n"
            "Control sequences comply with ASHRAE Guideline 36-2021 Section 5.16 for "
            "multi-zone VAV AHUs. Implemented sequences include:\n"
            "  • Supply air temperature reset based on zone demand\n"
            "  • Static pressure reset using trim-and-respond logic\n"
            "  • Economizer control with differential enthalpy switchover\n"
            "  • Morning warm-up and cool-down with optimum start\n"
            "  • Freeze protection with low-limit cutout at 35°F\n"
            "  • Smoke control mode (damper commands on alarm signal)",

            # COMPLIANT — VAV boxes
            "5.2  VAV Terminal Units\n"
            "Each VAV box controller implements ASHRAE Guideline 36 Section 5.8 sequences:\n"
            "  • Cooling-only VAV with reheat\n"
            "  • Zone temperature setpoint: 70°F occupied, 60°F unoccupied heating; "
            "75°F occupied, 85°F unoccupied cooling\n"
            "  • Minimum airflow setpoints per design documents",

            # PARTIAL — Chiller plant: sequences described but no specific ASHRAE 90.1 §6.5.4 compliance claim
            "5.3  Chiller Plant Sequencing\n"
            "The chiller plant control strategy includes lead/lag rotation among available "
            "chillers, staged loading based on chilled water supply temperature and "
            "differential pressure, and condenser water temperature reset. "
            "Variable speed primary chilled water pumps are controlled by differential "
            "pressure reset. The control logic is developed based on good engineering "
            "practice and manufacturer recommendations. Chiller integration is via "
            "BACnet/IP to the chiller manufacturer's gateway.",

            # NON_COMPLIANT — Spec requires Demand Controlled Ventilation (DCV) proof of CO2 reset
            # but proposal only says CO2 sensors are provided, not that DCV sequences are implemented
            "5.4  Ventilation and Indoor Air Quality\n"
            "CO2 sensors are installed in all high-occupancy spaces per the sensor "
            "schedule on drawing M-601. Outdoor air damper positions are controlled "
            "per the AHU sequences described in Section 5.1. "
            "Ventilation rates meet ASHRAE Standard 62.1-2022 minimum requirements.",

            # COMPLIANT — Hot water plant
            "5.5  Domestic Hot Water and Heating Hot Water\n"
            "Boiler lead/lag sequencing and outdoor air temperature reset of hot water "
            "supply temperature (180°F design, reset to 140°F at 60°F OAT). "
            "Condensing boiler efficiency optimization via flue gas temperature monitoring.",
        ],
    ),

    # -----------------------------------------------------------------------
    # 6. SYSTEM INTEGRATION
    # -----------------------------------------------------------------------
    (
        "6. System Integration",
        [
            # NON_COMPLIANT — Spec requires hard-wired dry contact interface to fire alarm
            # panel AND BACnet read of alarm points. Proposal only describes BACnet read.
            "6.1  Fire Alarm System Integration\n"
            "The BMS will interface with the Notifier NFS2-3030 fire alarm panel via "
            "BACnet/IP using the panel's optional BACnet gateway module (Notifier model "
            "BACnet-GW). The BMS will read fire alarm status points including: smoke "
            "detector activations, pull station activations, and system trouble status. "
            "Upon receipt of a fire alarm signal, the BMS will command AHU smoke "
            "control dampers to their designated positions.",

            # NOT_ADDRESSED — Spec requires BMS to command fire alarm panel auxiliary
            # relays for stairwell pressurization. This is completely omitted.

            # PARTIAL — Access control: read-only occupancy signal only, no bidirectional
            "6.2  Access Control Integration\n"
            "The BMS will receive occupancy status signals from the Lenel OnGuard access "
            "control system via a Modbus TCP interface. When the access control system "
            "reports a space as unoccupied for more than 30 minutes, the BMS will "
            "transition the affected zone to unoccupied setpoints. "
            "Write-back to the access control system is not included in this scope.",

            # COMPLIANT — Electrical metering
            "6.3  Electrical Sub-Metering\n"
            "Schneider Electric PowerLogic PM8000 series power meters are integrated via "
            "Modbus TCP. The BMS collects kW, kWh, kVAR, power factor, and voltage/current "
            "for each metered panel. Data is displayed on the energy dashboard and included "
            "in the monthly energy report.",

            # COMPLIANT — Lighting (basic relay control)
            "6.4  Lighting Control Integration\n"
            "Perimeter and parking lighting is controlled via BMS-controlled dry contact "
            "relays, scheduled by astronomical clock. Interior lighting control is handled "
            "by the separate Lutron GRAFIK Eye QS lighting control system; the BMS "
            "receives occupancy status from Lutron via BACnet/IP for energy reporting "
            "purposes only.",
        ],
    ),

    # -----------------------------------------------------------------------
    # 7. CYBERSECURITY
    # -----------------------------------------------------------------------
    (
        "7. Cybersecurity",
        [
            # PARTIAL — Mentions HTTPS and user auth but no mention of network segmentation,
            # patch management plan, or NIST CSF compliance as required by spec
            "7.1  Network Security\n"
            "The APEX Server web interface is served exclusively over HTTPS (TLS 1.2 or "
            "higher). The BMS VLAN is isolated from the general corporate network by a "
            "firewall; inbound connections from the internet are not permitted. "
            "Remote access for Nexus support is provided via an encrypted VPN tunnel "
            "that requires two-factor authentication.",

            "7.2  User Authentication and Authorization\n"
            "All operator access requires username and password authentication. "
            "Password policy enforces a minimum of 12 characters with complexity "
            "requirements, 90-day expiration, and lockout after 5 failed attempts. "
            "Role-based access control limits each user account to the minimum "
            "permissions required for their role.",

            # NON_COMPLIANT — Spec requires encrypted communications for all field
            # devices (BACnet/SC). Proposal uses standard BACnet/IP (unencrypted) at field level.
            "7.3  Field Network Security\n"
            "The field BACnet MS/TP and BACnet/IP networks operate on a dedicated "
            "VLAN with no direct internet exposure. Physical security of control panels "
            "is achieved via keyed enclosures. Field device firmware is updated during "
            "annual preventive maintenance visits.",

            # NOT_ADDRESSED — Spec requires a formal cybersecurity risk assessment
            # and written incident response plan. Neither is mentioned.
        ],
    ),

    # -----------------------------------------------------------------------
    # 8. ENERGY MANAGEMENT
    # -----------------------------------------------------------------------
    (
        "8. Energy Management and Reporting",
        [
            # COMPLIANT — Energy dashboard
            "8.1  Energy Dashboard\n"
            "The APEX Energy Dashboard provides real-time and historical views of "
            "whole-building and sub-metered energy consumption. Views include: "
            "kWh by system (HVAC, lighting, plug loads), energy intensity (kBtu/sqft), "
            "and comparative benchmarking against prior periods.",

            # PARTIAL — Monthly reports described but no automated ENERGY STAR upload
            # or ASHRAE 90.1 Energy Cost Budget reporting as required by spec
            "8.2  Automated Reporting\n"
            "Monthly energy reports are generated automatically on the first business "
            "day of each month and emailed to designated facility personnel. Reports "
            "include: consumption summary, peak demand analysis, and degree-day "
            "normalized comparison. Custom report templates are available.",

            # NOT_ADDRESSED — Spec requires demand response (DR) capability with
            # OpenADR 2.0 interface. This is completely omitted from proposal.
            "8.3  Load Shedding\n"
            "Manual load shedding sequences are implemented to reduce peak demand "
            "during high-tariff periods. The operator can activate preset load shed "
            "scenarios from the graphics interface. Automated load shedding based on "
            "real-time pricing signals is a future-phase option.",
        ],
    ),

    # -----------------------------------------------------------------------
    # 9. COMMISSIONING & TRAINING
    # -----------------------------------------------------------------------
    (
        "9. Commissioning, Training, and Documentation",
        [
            # PARTIAL — Functional testing described but no formal Cx agent witnessing
            # procedure or pre-functional checklists as required by spec
            "9.1  Commissioning\n"
            "Nexus will perform point-to-point verification for all hardwired I/O points "
            "and functional performance testing (FPT) of all control sequences. "
            "A commissioning report will be issued documenting test results. "
            "Nexus will coordinate with the commissioning authority (CxA) and provide "
            "access to all systems during CxA-witnessed tests.",

            # PARTIAL — Training offered but no minimum hour requirement stated;
            # spec requires 16 hours of on-site training
            "9.2  Training\n"
            "Nexus will provide operator training sessions for facility staff covering: "
            "system navigation, alarm response, setpoint adjustments, scheduling, "
            "and trend analysis. Training materials (user manual and quick reference "
            "cards) will be provided in digital and printed form. "
            "Training will be scheduled in coordination with the facility team.",

            # COMPLIANT — O&M documentation
            "9.3  Documentation\n"
            "The following documentation is included with the system delivery:\n"
            "  • As-built control drawings (AutoCAD and PDF)\n"
            "  • Sequence of operations narrative\n"
            "  • Point list and I/O schedule\n"
            "  • Controller configuration backup files\n"
            "  • Maintenance and troubleshooting manual\n"
            "  • Warranty documentation",

            "9.4  Warranty\n"
            "Nexus provides a 24-month warranty on all supplied equipment and software "
            "from the date of substantial completion. The warranty covers defects in "
            "materials and workmanship. Response time for critical failures: 4 hours. "
            "Annual preventive maintenance visits are included during the warranty period.",
        ],
    ),

    # -----------------------------------------------------------------------
    # 10. STANDARDS COMPLIANCE MATRIX
    # -----------------------------------------------------------------------
    (
        "10. Standards Compliance Summary",
        [
            "The Nexus APEX system is designed to comply with the following standards "
            "and guidelines:\n\n"
            "ANSI/ASHRAE Standard 135-2020 (BACnet):\n"
            "  All controllers are BTL-certified. Conformance classes as noted in "
            "Section 2. [COMPLIANT]\n\n"
            "ASHRAE Guideline 36-2021 (High-Performance Sequences):\n"
            "  AHU and VAV sequences implemented per GL-36. Chiller plant sequences "
            "based on GL-36 principles with manufacturer-specific adaptations. [PARTIAL]\n\n"
            "ASHRAE Standard 62.1-2022 (Ventilation):\n"
            "  Minimum outdoor air rates programmed per design engineer's schedule. "
            "DCV implementation per Section 5.4 of this submittal. [SEE NOTE]\n\n"
            "ASHRAE Standard 90.1-2022 (Energy Efficiency):\n"
            "  Variable speed drives on all HVAC fans and pumps >1 HP. Supply air "
            "temperature reset, static pressure reset, and hot water reset implemented. "
            "Demand response capability not included in base scope. [PARTIAL]\n\n"
            "NFPA 72-2022 (Fire Alarm):\n"
            "  BMS interface to fire alarm system per Section 6.1. "
            "Hard-wired supervisory circuit not included; BACnet gateway integration "
            "only. [SEE NOTE — may require CxA review]\n\n"
            "UL 916 (Energy Management Equipment):\n"
            "  All DDC controllers carry UL 916 listing. [COMPLIANT]\n\n"
            "IEC 62443 / NIST Cybersecurity Framework:\n"
            "  Perimeter security measures implemented per Section 7. Formal risk "
            "assessment not included in scope. [PARTIAL]",
        ],
    ),

    # -----------------------------------------------------------------------
    # 11. PRODUCT DATA SHEETS (abbreviated)
    # -----------------------------------------------------------------------
    (
        "11. Product Data — NX-4000 Building Automation Controller",
        [
            "Manufacturer: Nexus Controls & Automation, LLC\n"
            "Model: NX-4000-IP\n"
            "Firmware Version: 4.12.3\n\n"
            "Technical Specifications:\n"
            "  Operating Temperature: 32°F to 122°F (0°C to 50°C)\n"
            "  Storage Temperature: -40°F to 158°F (-40°C to 70°C)\n"
            "  Humidity: 5% to 95% RH non-condensing\n"
            "  Power Consumption: 15W maximum\n"
            "  Dimensions: 12.6\" × 7.8\" × 3.5\" (DIN rail mount)\n"
            "  Weight: 2.1 lbs\n"
            "  Certifications: UL 916, CE, FCC Part 15 Class B, BTL-listed BACnet/IP B-BC\n\n"
            "I/O Capacity (base unit, expandable):\n"
            "  Universal Inputs (UI): 8 — configurable as 0-10V, 4-20mA, resistive, or DI\n"
            "  Analog Outputs (AO): 4 — 0-10 VDC, 20 mA max sink\n"
            "  Digital Outputs (DO): 6 — TRIAC, 24 VAC, 0.5A max\n"
            "  MS/TP Ports: 2 (field bus)\n"
            "  Ethernet Ports: 1 × RJ-45, 10/100/1000BASE-T",

            "Memory and Processing:\n"
            "  Program Memory: 256 MB Flash\n"
            "  RAM: 128 MB (battery-backed SRAM for setpoints and schedules)\n"
            "  Real-Time Clock: Battery-backed, ±1 second/day accuracy\n"
            "  Data Retention on Power Loss: All setpoints, schedules, and trend data "
            "retained for minimum 72 hours via internal UPS capacitor",
        ],
    ),
]


# ---------------------------------------------------------------------------
# PDF generation
# ---------------------------------------------------------------------------

class BmsProposalPDF(FPDF):
    """Custom FPDF class for the BMS proposal."""

    BRAND_BLUE = (0, 62, 114)
    BRAND_GRAY = (100, 100, 100)
    LIGHT_BLUE = (220, 235, 248)

    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(*self.BRAND_GRAY)
        self.cell(0, 6, "Nexus Controls & Automation, LLC  |  BMS Technical Submittal", align="L")
        self.cell(0, 6, f"Page {self.page_no()}", align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*self.BRAND_BLUE)
        self.set_line_width(0.3)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(3)
        self.set_text_color(0, 0, 0)

    def footer(self):
        if self.page_no() == 1:
            return
        self.set_y(-12)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(*self.BRAND_GRAY)
        self.cell(0, 5,
                  _ascii("CONFIDENTIAL -- For review by designated recipient only. "
                  "Synthetic document generated for compliance testing purposes."),
                  align="C")
        self.set_text_color(0, 0, 0)

    def cover_page(self, cover: dict) -> None:
        self.add_page()
        # Top accent bar
        self.set_fill_color(*self.BRAND_BLUE)
        self.rect(0, 0, self.w, 28, "F")

        # Company name in white
        self.set_y(8)
        self.set_font("Helvetica", "B", 18)
        self.set_text_color(255, 255, 255)
        self.cell(0, 10, cover["company"], align="C", new_x="LMARGIN", new_y="NEXT")

        # Divider
        self.ln(30)
        self.set_draw_color(*self.BRAND_BLUE)
        self.set_line_width(1.0)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(6)

        # Title block
        self.set_font("Helvetica", "B", 22)
        self.set_text_color(*self.BRAND_BLUE)
        self.multi_cell(0, 12, "TECHNICAL SUBMITTAL\nBuilding Management System", align="C")
        self.ln(4)

        self.set_font("Helvetica", "", 14)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 8, _ascii(cover["project"]), align="C")
        self.ln(10)

        # Info table
        self.set_line_width(0.3)
        info_rows = [
            ("Submittal No.:", cover["submittal"]),
            ("Date:", cover["date"]),
            ("Prepared by:", cover["prepared_by"]),
            ("Submitted to:", cover["submitted_to"]),
        ]
        col_w = (self.w - self.l_margin - self.r_margin) / 2
        for label, value in info_rows:
            self.set_font("Helvetica", "B", 10)
            self.set_text_color(*self.BRAND_GRAY)
            self.cell(col_w, 9, _ascii(label), align="R")
            self.set_font("Helvetica", "", 10)
            self.set_text_color(0, 0, 0)
            self.cell(col_w, 9, _ascii(value), align="L", new_x="LMARGIN", new_y="NEXT")

        self.ln(14)
        # Bottom accent bar
        self.set_fill_color(*self.LIGHT_BLUE)
        self.set_y(self.h - 40)
        self.rect(0, self.h - 38, self.w, 38, "F")
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(*self.BRAND_GRAY)
        self.set_y(self.h - 30)
        self.multi_cell(0, 6,
            _ascii("This document contains proprietary information. Distribution limited to "
            "designated project personnel.\n"
            "SYNTHETIC DOCUMENT -- Generated for compliance testing purposes only."),
            align="C")
        self.set_text_color(0, 0, 0)

    def toc_page(self, sections: list[tuple[str, list[str]]]) -> None:
        self.add_page()
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(*self.BRAND_BLUE)
        self.cell(0, 10, "Table of Contents", new_x="LMARGIN", new_y="NEXT")
        self.ln(4)
        self.set_draw_color(*self.BRAND_BLUE)
        self.set_line_width(0.5)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(6)

        for i, (title, _) in enumerate(sections, 1):
            self.set_font("Helvetica", "", 11)
            self.set_text_color(40, 40, 40)
            dots = "." * max(2, 60 - len(title))
            self.cell(0, 8, _ascii(f"{title}  {dots}  {i + 2}"), new_x="LMARGIN", new_y="NEXT")

    def section_page(self, title: str, paragraphs: list[str]) -> None:
        self.add_page()
        # Section title bar
        self.set_fill_color(*self.LIGHT_BLUE)
        self.rect(self.l_margin - 2, self.get_y() - 2,
                  self.w - self.l_margin - self.r_margin + 4, 14, "F")
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(*self.BRAND_BLUE)
        self.cell(0, 10, _ascii(title), new_x="LMARGIN", new_y="NEXT")
        self.ln(6)
        self.set_text_color(0, 0, 0)

        body_w = self.w - self.l_margin - self.r_margin
        indent = 6  # mm for bullet lines

        for para in paragraphs:
            lines = _ascii(para).split("\n")
            for j, line in enumerate(lines):
                self.set_x(self.l_margin)  # always reset x
                stripped = line.strip()
                if not stripped:
                    self.ln(2)
                    continue
                if j == 0 and stripped.endswith((")", ":")):
                    # Sub-heading
                    self.set_font("Helvetica", "B", 10)
                    self.set_text_color(*self.BRAND_BLUE)
                    self.multi_cell(body_w, 6, stripped)
                    self.set_font("Helvetica", "", 10)
                    self.set_text_color(0, 0, 0)
                elif stripped.startswith("- ") or stripped.startswith("-  "):
                    self.set_x(self.l_margin + indent)
                    self.multi_cell(body_w - indent, 5.5, stripped)
                else:
                    self.set_font("Helvetica", "", 10)
                    self.multi_cell(body_w, 5.5, stripped)
            self.ln(4)


def build_pdf(cover: dict, sections: list[tuple[str, list[str]]], output: Path) -> None:
    pdf = BmsProposalPDF(orientation="P", unit="mm", format="Letter")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(left=20, top=20, right=20)

    pdf.cover_page(cover)
    pdf.toc_page(sections)

    for title, paragraphs in sections:
        pdf.section_page(title, paragraphs)

    output.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(output))
    print(f"Generated: {output}  ({output.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    build_pdf(COVER, SECTIONS, OUTPUT)
