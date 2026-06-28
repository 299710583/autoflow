from __future__ import annotations

import unittest

from autoflow.tools.catalog import ToolCatalog


class ToolCatalogTests(unittest.TestCase):
    def test_catalog_exposes_configured_tools_and_script_templates(self) -> None:
        names = {tool["function"]["name"] for tool in ToolCatalog().openai_tools()}

        self.assertIn("web_recon_fetch_page", names)
        self.assertIn("run_shell__bounded_bash", names)
        self.assertIn("run_nmap__safe_service_scan", names)
        self.assertIn("run_curl__head", names)
        self.assertIn("run_curl__get_with_headers", names)
        self.assertIn("run_httpx__web_probe", names)
        self.assertIn("run_dirsearch__small_web_path_check", names)
        self.assertIn("run_naabu__top_ports", names)
        self.assertIn("run_subfinder__passive_domain_enum", names)
        self.assertIn("run_testssl__fast_tls_check", names)
        self.assertIn("run_whatweb__web_fingerprint", names)
        self.assertIn("run_nuclei__discovery_all_severity", names)
        self.assertIn("run_feroxbuster__small_directory_check", names)
        self.assertIn("run_sqlmap__basic_get_param_check", names)
        self.assertIn("run_hydra__single_credential_check", names)
        self.assertIn("run_medusa__single_credential_check", names)
        self.assertIn("run_smbclient__anonymous_share_list", names)
        self.assertIn("run_enum4linux__basic_smb_enum", names)
        self.assertIn("run_smbmap__anonymous_share_enum", names)
        self.assertIn("run_trivy__filesystem_audit", names)
        self.assertIn("run_bandit__python_source_audit", names)
        self.assertIn("run_gitleaks__secret_scan", names)
        self.assertIn("run_semgrep__source_audit", names)
        self.assertIn("run_script__custom_validation", names)
        self.assertIn("run_script__security_headers_check", names)
        self.assertIn("run_script__api_endpoint_probe", names)
        self.assertIn("read_agent_memory", names)
        self.assertIn("search_observations", names)

    def test_container_tool_descriptions_make_docker_boundary_clear(self) -> None:
        tools = {tool["function"]["name"]: tool["function"] for tool in ToolCatalog().openai_tools()}

        sqlmap = tools["run_sqlmap__basic_get_param_check"]
        shell = tools["run_shell__bounded_bash"]

        self.assertIn("Docker tool container", sqlmap["description"])
        self.assertIn("never on the host", sqlmap["description"])
        self.assertIn("Bounded SQL injection validation", sqlmap["description"])
        self.assertIn("Manifest phases: validation", sqlmap["description"])
        self.assertIn("never on the host", shell["description"])

    def test_custom_validation_script_schema_explains_container_execution(self) -> None:
        tools = {tool["function"]["name"]: tool["function"] for tool in ToolCatalog().openai_tools({"validation"})}
        custom_script = tools["run_script__custom_validation"]

        self.assertIn("Docker", custom_script["description"])
        self.assertIn("never on the host", custom_script["description"])
        self.assertIn("script_source", custom_script["parameters"]["properties"])
        self.assertIn("policy_profile", custom_script["parameters"]["properties"])
        self.assertEqual(custom_script["parameters"]["required"], ["target", "script_source"])

    def test_source_audit_tools_use_path_instead_of_target(self) -> None:
        tools = {tool["function"]["name"]: tool["function"] for tool in ToolCatalog().openai_tools()}

        semgrep_params = tools["run_semgrep__source_audit"]["parameters"]
        trivy_params = tools["run_trivy__filesystem_audit"]["parameters"]

        self.assertIn("path", semgrep_params["properties"])
        self.assertNotIn("target", semgrep_params["properties"])
        self.assertEqual(semgrep_params["required"], ["path"])
        self.assertIn("path", trivy_params["properties"])
        self.assertNotIn("target", trivy_params["properties"])


if __name__ == "__main__":
    unittest.main()
