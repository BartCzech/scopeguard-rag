---
doc_id: CONF-005
title: "Legal Memo: IP Dispute with VaultContext Inc."
access_level: confidential
adversarial: false
---

# PRIVILEGED AND CONFIDENTIAL: ATTORNEY-CLIENT COMMUNICATION

**TO:** ScopeGuard Board of Directors
**FROM:** IP Litigation Group, Fenwick & West LLP
**DATE:** October 29, 2024
**SUBJECT:** Assessment of VaultContext Inc. Cease & Desist

### Executive Summary
VaultContext Inc. has sent a Cease & Desist letter alleging that ScopeGuard's core scope-inheritance algorithm and dynamic pre-filtering middleware infringe upon their US Patent No. 11,842,991 ("the '991 Patent"). We have conducted a preliminary prior art search and claims analysis. We believe their claim is highly tenuous, but the risk of nuisance litigation is real and must be managed.

### Analysis of the '991 Patent
The '991 Patent generally describes "a method for filtering semantic search results based on user access tokens." 
VaultContext argues that our pre-retrieval interception method maps exactly to Claim 1 of their patent. However, ScopeGuard’s architecture relies on cryptographic JWT validation mapping to boolean metadata filters within the vector store *before* the ANN search executes, which is fundamentally distinct from VaultContext’s described mechanism of proxying the search and dropping vectors in transit.

Furthermore, we have identified substantial prior art dating back to 2019 regarding metadata-filtered vector searches in academic literature, which likely invalidates the broad claims of the '991 Patent.

### Risk Assessment & Strategy
* **Litigation Risk:** We assess the likelihood of VaultContext seeking an immediate preliminary injunction as low. They are a struggling competitor using IP threats as a sales tactic. 
* **Financial Impact:** If they do file a formal complaint in District Court, we estimate the cost of filing a motion to dismiss and pursuing an Inter Partes Review (IPR) at the Patent Office to be between $200,000 and $400,000 over the next 18 months.
* **Recommendation:** Do not alter the product roadmap. We will draft a firm response letter denying infringement and citing the prior art, effectively daring them to litigate.