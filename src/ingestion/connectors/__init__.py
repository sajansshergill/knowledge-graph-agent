"""
connectors/
-----------
Source connectors for the EKGA ingestion layer.

Each connector is fully self-contained (no shared base class).
All return typed dataclasses with a .to_dict() method for
downstream compatibility with the chunker and graph loader.

Connectors
----------
ConfluenceConnector  — Confluence Cloud REST API v2
SlackConnector       — Slack export ZIP or Web API
PDFConnector         — Local directory or GCS bucket
JiraConnector        — Jira Cloud REST API v3
"""

from .confluence_connector import ConfluenceConnector, ConfluencePage
from .slack_connector import SlackConnector, SlackThread, SlackMessage
from .pdf_connector import PDFConnector, PDFDocument, PDFPage
from .jira_connector import JiraConnector, JiraIssue, JiraComment, JiraLink

__all__ = [
    # Confluence
    "ConfluenceConnector",
    "ConfluencePage",
    # Slack
    "SlackConnector",
    "SlackThread",
    "SlackMessage",
    # PDF
    "PDFConnector",
    "PDFDocument",
    "PDFPage",
    # Jira
    "JiraConnector",
    "JiraIssue",
    "JiraComment",
    "JiraLink",
]