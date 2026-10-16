-- Run in Supabase: SQL Editor -> New query -> paste -> Run.

create table public.workspaces (
  id uuid primary key default gen_random_uuid(),
  name text not null,
  created_by uuid not null default auth.uid() references auth.users(id),
  created_at timestamptz not null default now()
);

create table public.workspace_members (
  workspace_id uuid not null references public.workspaces(id) on delete cascade,
  user_id uuid not null references auth.users(id) on delete cascade,
  role text not null default 'agent' check (role in ('owner','admin','agent')),
  primary key (workspace_id, user_id)
);

create table public.documents (
  id uuid primary key default gen_random_uuid(),
  workspace_id uuid not null references public.workspaces(id) on delete cascade,
  filename text not null,
  status text not null default 'processing' check (status in ('processing','ready','failed')),
  created_at timestamptz not null default now()
);

create table public.tickets (
  id uuid primary key default gen_random_uuid(),
  workspace_id uuid not null references public.workspaces(id) on delete cascade,
  message text not null,
  category text, priority text,
  status text not null default 'draft_ready' check (status in ('draft_ready','needs_review','reviewed')),
  created_at timestamptz not null default now()
);

create table public.drafts (
  id uuid primary key default gen_random_uuid(),
  workspace_id uuid not null references public.workspaces(id) on delete cascade,
  ticket_id uuid not null references public.tickets(id) on delete cascade,
  body text not null,
  needs_human boolean not null,
  reviewed_by uuid references auth.users(id),
  created_at timestamptz not null default now()
);

create table public.feedback (
  id uuid primary key default gen_random_uuid(),
  workspace_id uuid not null references public.workspaces(id) on delete cascade,
  ticket_id uuid not null references public.tickets(id) on delete cascade,
  rating text not null check (rating in ('useful','incorrect','needs_escalation')),
  correction text,
  allow_training_use boolean not null default false,
  created_at timestamptz not null default now()
);

-- Membership check used by every policy. SECURITY DEFINER avoids recursive RLS on workspace_members.
create or replace function public.is_member(ws uuid) returns boolean
language sql security definer stable set search_path = public as $$
  select exists (select 1 from public.workspace_members m where m.workspace_id = ws and m.user_id = auth.uid());
$$;

-- The creator of a workspace automatically becomes its owner.
create or replace function public.add_owner() returns trigger
language plpgsql security definer set search_path = public as $$
begin
  insert into public.workspace_members (workspace_id, user_id, role) values (new.id, new.created_by, 'owner');
  return new;
end $$;
create trigger workspaces_add_owner after insert on public.workspaces
  for each row execute function public.add_owner();

-- Row Level Security: ON for every table, then policies.
alter table public.workspaces enable row level security;
alter table public.workspace_members enable row level security;
alter table public.documents enable row level security;
alter table public.tickets enable row level security;
alter table public.drafts enable row level security;
alter table public.feedback enable row level security;

create policy ws_select on public.workspaces for select using (public.is_member(id));
create policy ws_insert on public.workspaces for insert with check (created_by = auth.uid());
create policy mem_select on public.workspace_members for select using (public.is_member(workspace_id));

create policy docs_all on public.documents for all using (public.is_member(workspace_id)) with check (public.is_member(workspace_id));
create policy tickets_all on public.tickets for all using (public.is_member(workspace_id)) with check (public.is_member(workspace_id));
create policy drafts_all on public.drafts for all using (public.is_member(workspace_id)) with check (public.is_member(workspace_id));
create policy feedback_all on public.feedback for all using (public.is_member(workspace_id)) with check (public.is_member(workspace_id));

create index on public.tickets (workspace_id, created_at desc);
create index on public.drafts (ticket_id);
