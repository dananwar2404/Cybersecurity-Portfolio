# Equifax Breach Threat Analysis & MITRE ATT&CK Mapping

## Project Overview

This project analyzes the 2017 Equifax data breach using the MITRE ATT&CK framework. The investigation reconstructs the attack lifecycle, maps observed and likely adversary tactics, techniques, and procedures, and translates those findings into defensive detection and mitigation recommendations.

The breach began when attackers exploited an unpatched Apache Struts vulnerability in Equifax's online dispute portal. After gaining access, the attackers installed web shells, discovered exposed credentials, accessed additional databases, collected sensitive information, and exfiltrated large volumes of personally identifiable information.

---

## Project Objectives

- Reconstruct the Equifax breach attack lifecycle
- Identify attacker tactics, techniques, and procedures
- Map attacker behavior to MITRE ATT&CK
- Visualize the intrusion using ATT&CK Navigator
- Identify detection opportunities
- Develop mitigation and security-hardening recommendations
- Produce a professional threat intelligence report

---

## Skills Demonstrated

- Threat Intelligence Analysis
- MITRE ATT&CK Mapping
- ATT&CK Navigator
- Adversary Behavior Analysis
- Incident Reconstruction
- Detection Engineering
- Security Control Development
- Technical Security Reporting

---

## Tools and Frameworks

- MITRE ATT&CK
- MITRE ATT&CK Navigator
- Microsoft Excel
- Public Threat Intelligence Sources
- Congressional Breach Reporting
- Security Incident Analysis

---

## Executive Summary

The Equifax breach was a prolonged data-theft operation enabled by an unpatched, internet-facing Apache Struts vulnerability. After compromising the public dispute portal, the attackers installed approximately 30 web shells and located unencrypted credentials that allowed them to access additional systems and databases.

The attackers reportedly accessed 48 unrelated databases, issued approximately 9,000 queries, and conducted 265 successful database searches before compressing and exfiltrating sensitive personal information.

The intrusion remained undetected for 76 days partly because an expired security certificate prevented a network-monitoring device from inspecting encrypted traffic.

---

## Attack Lifecycle

| Attack Phase | Observed or Likely Activity |
|---|---|
| Reconnaissance | Scanning public-facing systems and identifying vulnerable software |
| Initial Access | Exploitation of the Apache Struts web application |
| Execution | Injection of malicious commands through the vulnerable application |
| Persistence | Installation and use of web shells |
| Defense Evasion | Exploitation of weaknesses to avoid security controls |
| Credential Access | Discovery of plaintext credentials stored in files |
| Discovery | Identification of internal systems, accounts, files, and databases |
| Lateral Movement | Access to additional systems and databases |
| Collection | Large-scale database queries and collection of sensitive records |
| Command and Control | Use of common web and application-layer protocols |
| Exfiltration | Compression and transfer of stolen data through web-based channels |

---

## Key MITRE ATT&CK Techniques

### Reconnaissance

- **T1595.002 – Vulnerability Scanning**
- **T1592.002 – Software**
- **T1596.005 – Scan Databases**
- **T1594 – Search Victim-Owned Websites**

### Initial Access

- **T1190 – Exploit Public-Facing Application**
- **T1133 – External Remote Services**

### Execution

- **T1674 – Input Injection**

### Persistence

- **T1505.003 – Web Shell**
- **T1133 – External Remote Services**

### Credential Access

- **T1552.001 – Credentials in Files**

### Discovery

- **T1083 – File and Directory Discovery**
- **T1018 – Remote System Discovery**
- **T1087.001 – Local Account**
- **T1213.006 – Databases**

### Lateral Movement

- **T1210 – Exploitation of Remote Services**

### Collection

- **T1213.006 – Databases**
- **T1119 – Automated Collection**

### Command and Control

- **T1071.001 – Web Protocols**
- **T1071.002 – File Transfer Protocols**
- **T1102.002 – Bidirectional Communication**

### Exfiltration

- **T1567 – Exfiltration Over Web Service**

---

## Detection Opportunities

The analysis identified several opportunities for earlier detection:

- Monitor perimeter firewall and IDS logs for repeated probing and vulnerability scanning.
- Alert on suspicious requests and injection patterns targeting public-facing applications.
- Monitor web directories for unauthorized scripts and web-shell activity.
- Review authentication logs for access from unfamiliar IP addresses or locations.
- Detect internal scanning or enumeration originating from public-facing servers.
- Monitor database activity for abnormal query volume, unusual accounts, and large result sets.
- Inspect outbound traffic for unusual file transfers, rare destinations, and unexpected protocols.
- Alert on compressed or archived data leaving systems that contain sensitive information.
- Monitor certificates and security appliances to ensure monitoring controls remain operational.

---

## Defensive Recommendations

### Patch Management

Establish an enforced process for rapidly identifying and patching critical vulnerabilities affecting internet-facing systems.

### Web Application Security

Deploy web application firewall protections and monitor application logs for suspicious requests, injection attempts, and abnormal server activity.

### Credential Protection

Remove plaintext credentials from files and shared locations. Store credentials using approved secrets-management systems and rotate exposed credentials.

### Network Segmentation

Separate public-facing systems from internal databases and other systems containing regulated or sensitive information.

### Database Monitoring

Implement database activity monitoring to detect unusual query patterns, excessive record access, and large data exports.

### Certificate Management

Maintain a centralized inventory of digital certificates, configure expiration alerts, and automate certificate renewal where possible.

### Security Monitoring

Validate that IDS, firewall, proxy, endpoint, and network-monitoring systems are operational and successfully collecting security telemetry.

---

## Key Lessons Learned

The Equifax breach demonstrates how several security weaknesses can combine into a major incident. A single unpatched vulnerability enabled initial access, while weak credential storage and insufficient segmentation allowed attackers to move beyond the compromised web server.

The incident also demonstrates that security controls must be continuously validated. A monitoring solution provides little protection when certificate expiration or configuration failures prevent it from inspecting traffic.

The mapped attack chain shows how Initial Access, Persistence, Credential Access, Discovery, Lateral Movement, Collection, Command and Control, and Exfiltration can connect during a long-term data-theft campaign.

---

## Project Deliverables

- MITRE ATT&CK Navigator visualization
- ATT&CK tactics and techniques mapping
- Incident debrief
- Detection recommendations
- Mitigation and security-hardening recommendations
- Five-page technical report

---

## Full Report

[View the complete Equifax MITRE ATT&CK Mapping Report](Threat%20Intelligence/Anwar_Fanco%20MitreAttackwk10.pdf ':ignore')

---
