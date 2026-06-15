from __future__ import annotations

from pathlib import Path
from typing import Any
from xml.etree import ElementTree


def parse_nmap_xml(path: str | Path) -> list[dict[str, Any]]:
    xml_path = Path(path)
    tree = ElementTree.parse(xml_path)
    return parse_nmap_xml_text(ElementTree.tostring(tree.getroot(), encoding="unicode"))


def parse_nmap_xml_text(xml_text: str) -> list[dict[str, Any]]:
    root = ElementTree.fromstring(xml_text)
    assets: list[dict[str, Any]] = []

    for host in root.findall("host"):
        host_state = host.find("status")
        if host_state is not None and host_state.get("state") != "up":
            continue

        address = _first_address(host)
        hostname = _first_hostname(host)
        ports = []

        ports_node = host.find("ports")
        if ports_node is not None:
            for port_node in ports_node.findall("port"):
                state_node = port_node.find("state")
                state = state_node.get("state", "") if state_node is not None else ""
                if state != "open":
                    continue

                service_node = port_node.find("service")
                ports.append(
                    {
                        "port": int(port_node.get("portid", "0")),
                        "protocol": port_node.get("protocol", ""),
                        "state": state,
                        "service": service_node.get("name", "") if service_node is not None else "",
                        "product": service_node.get("product", "") if service_node is not None else "",
                        "version": service_node.get("version", "") if service_node is not None else "",
                        "extrainfo": service_node.get("extrainfo", "") if service_node is not None else "",
                    }
                )

        assets.append(
            {
                "ip": address,
                "hostname": hostname,
                "ports": ports,
            }
        )

    return assets


def _first_address(host: ElementTree.Element) -> str:
    for address in host.findall("address"):
        if address.get("addrtype") in {"ipv4", "ipv6"}:
            return address.get("addr", "")
    address = host.find("address")
    return address.get("addr", "") if address is not None else ""


def _first_hostname(host: ElementTree.Element) -> str:
    hostnames = host.find("hostnames")
    if hostnames is None:
        return ""
    hostname = hostnames.find("hostname")
    return hostname.get("name", "") if hostname is not None else ""
