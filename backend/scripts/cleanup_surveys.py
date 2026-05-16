"""List surveys and delete failed/partial ones to allow clean re-upload."""
import asyncio
import os
from pathlib import Path

for line in (Path(__file__).parent.parent.parent / ".env").read_text().splitlines():
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


async def main() -> None:
    from supabase._async.client import create_client
    db = await create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])

    surveys = (await db.table("surveys").select("id, name, file_name").execute()).data or []
    jobs = (await db.table("ingestion_jobs").select("survey_id, stage, status, last_error").execute()).data or []
    job_map = {j["survey_id"]: j for j in jobs}

    print(f"{'ID':36}  {'Name':45}  Stage/Status")
    print("-" * 100)
    to_delete = []
    for s in surveys:
        j = job_map.get(s["id"], {})
        stage = j.get("stage", "—")
        status = j.get("status", "—")
        marker = " [DELETE]" if status in ("failed", "pending") and stage != "done" else ""
        print(f"{s['id']}  {s['name'][:45]:45}  {stage}/{status}{marker}")
        if marker:
            to_delete.append(s["id"])

    if not to_delete:
        print("\nNothing to delete.")
        return

    print(f"\nDeleting {len(to_delete)} surveys with failed/stuck jobs...")
    for sid in to_delete:
        # Clean up wiki_pages
        wiki = (await db.table("wiki_pages").select("id, survey_ids").execute()).data or []
        for page in wiki:
            ids = [str(i) for i in (page.get("survey_ids") or [])]
            if sid not in ids:
                continue
            remaining = [i for i in ids if i != sid]
            if not remaining:
                await db.table("wiki_pages").delete().eq("id", page["id"]).execute()
            else:
                await db.table("wiki_pages").update({"survey_ids": remaining}).eq("id", page["id"]).execute()
        # Clean wiki_log
        await db.table("wiki_log").delete().eq("survey_id", sid).execute()
        # Delete survey (cascades everything else)
        await db.table("surveys").delete().eq("id", sid).execute()
        print(f"  Deleted {sid}")

    print("Done.")


asyncio.run(main())
