"""Workspace-local entry point for the Kaggle CLI."""
import sys
from pathlib import Path
from pip._vendor import truststore
truststore.inject_into_ssl()
sys.path.insert(0, str(Path(__file__).resolve().parent / '.kaggle-cli'))
from kaggle.cli import main

if __name__ == '__main__':
    if sys.argv[1:] == ['auth', 'login']:
        # Kaggle 2.2.4 authenticates before dispatching its login subcommand.
        # Call its login API directly and keep callback codes out of HTTP logs.
        from http.server import BaseHTTPRequestHandler
        BaseHTTPRequestHandler.log_message = lambda *args: None
        from kaggle import api
        api.auth_login_cli()
    else:
        main()
