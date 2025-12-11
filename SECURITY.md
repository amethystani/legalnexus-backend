# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in LegalNexus, please report it responsibly:

1. **Do not** open a public issue
2. Email the maintainers directly at: animesh.sinha@snu.edu.in
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Any suggested fixes

## Response Timeline

- **Acknowledgment**: Within 48 hours
- **Initial Assessment**: Within 1 week
- **Resolution**: Depends on severity

## Supported Versions

| Version | Supported |
|---------|-----------|
| main branch | Yes |
| Other branches | No |

## Security Best Practices

When using LegalNexus:

1. **Never commit secrets**: Use `config/.env` for API keys (gitignored)
2. **Keep dependencies updated**: Run `pip install --upgrade -r config/requirements.txt`
3. **Use virtual environments**: Isolate project dependencies
4. **Review third-party code**: Before running scripts from `scripts/`

## Known Security Considerations

- Embedding files (`.pkl`) are loaded with pickle - only use trusted files
- Neo4j credentials should be stored in environment variables
- API keys should never be committed to version control
