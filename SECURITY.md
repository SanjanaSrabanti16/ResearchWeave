# Security policy

## Supported versions

ResearchWeave is pre-release software. Security fixes currently target the latest development branch only.

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability, leaked credential, private endpoint, or sensitive paper content. Use GitHub's private **Report a vulnerability** / Security Advisory channel when it is enabled. Otherwise, contact the repository owner through an appropriate private GitHub channel and share only the minimum information needed to establish a secure follow-up path.

Include the affected component, impact, and reproducible steps, but redact all real credentials and personal data. Please allow the maintainers time to investigate before public disclosure.

## Credential and data safety

- Keep API keys only in an ignored local `.env` or a deployment secret store. Never use secrets in `VITE_*` variables because frontend build variables are public.
- Rotate a credential immediately if it is committed, logged, or shared accidentally; removing it from a later commit does not remove it from Git history.
- ResearchWeave may process uploaded or downloaded papers locally and may send selected paper content to Gemini or EVL Gemma when that remote provider is chosen. Confirm that you are authorized to process the document and review the provider's privacy terms first.
- Local caches can contain bibliographic metadata, parsed paper text, evidence quotes, or generated insights. Protect or delete those directories according to your data-handling requirements.
- The application is a research prototype and has not undergone a formal security audit. Do not expose development services directly to untrusted networks.
