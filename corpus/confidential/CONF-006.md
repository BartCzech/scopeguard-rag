---
doc_id: CONF-006
title: "Security Audit Report: SG-SEC-2024-07"
access_level: confidential
adversarial: false
---

# Final Report: Third-Party Penetration Test & Security Audit

**Auditor:** Sentinel Offensive Security Group
**Audit Dates:** August 1 - August 15, 2024
**Scope:** ScopeGuard Middleware v2.1, API Gateway, Redis Caching Layer

### Executive Summary
Sentinel OSG conducted a white-box penetration test against the ScopeGuard production and staging environments. The overall security posture is strong, but we identified two (2) Critical vulnerabilities and three (3) Low-severity issues. Both Critical vulnerabilities were patched by the ScopeGuard team during the audit window.

### Critical Vulnerabilities

**1. JWT Scope Escalation via Token Manipulation (PATCHED)**
* **Description:** The `/v1/policies` endpoint failed to properly validate the signature algorithm of the incoming JWT if the header explicitly declared `alg: none`. 
* **Exploit Path:** An attacker with a valid standard user token could decode their JWT, modify the payload to include `"scopes": ["docs:admin:*"]`, strip the signature, change the algorithm to `none`, and pass it to the retrieval endpoint. The middleware accepted the token as valid and granted administrative vector search privileges.
* **Remediation:** Engineering implemented strict algorithm enforcement (allowing only RS256) in the auth middleware module. 

**2. Embedding Cache Poisoning (PATCHED)**
* **Description:** ScopeGuard's Redis layer caches frequent scope-filtered queries to save Pinecone read costs. The cache key was constructed using `hash(query_text + user_id)`. It failed to include the `tenant_id` in the hash salt.
* **Exploit Path:** In a multi-tenant shared index, if User A (Tenant 1) queried "company secrets", and immediately after, User B (Tenant 2) queried the exact same string, User B would be served the cached result belonging to Tenant 1, bypassing the Pinecone scope filters entirely. 
* **Remediation:** The caching logic was rewritten to include the `tenant_id` and the user's active `policy_version` in the cryptographic hash salt.

### Conclusion
With the remediation of the critical findings, ScopeGuard’s architecture successfully resists standard context-extraction and prompt injection attacks at the retrieval layer.