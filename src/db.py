"""Starting sketch for Supabase persistence (not wired into app.py yet).
pip install supabase ; set SUPABASE_URL and SUPABASE_ANON_KEY (anon key only - never the service-role key here)."""
import os

from supabase import create_client


def sign_in(email: str, password: str):
    client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_ANON_KEY"])
    session = client.auth.sign_in_with_password({"email": email, "password": password}).session
    return client, session.user.id


def create_workspace(client, name: str):
    return client.table("workspaces").insert({"name": name}).execute().data[0]["id"]


def save_ticket(client, workspace_id: str, user_id: str, question: str, result: dict):
    status = "needs_review" if result["needs_human"] else "draft_ready"
    ticket = client.table("tickets").insert({
        "workspace_id": workspace_id, "message": question,
        "category": result["category"], "priority": result["priority"], "status": status,
    }).execute().data[0]
    client.table("drafts").insert({
        "workspace_id": workspace_id, "ticket_id": ticket["id"],
        "body": result["draft_reply"], "needs_human": result["needs_human"],
    }).execute()
    return ticket["id"]
