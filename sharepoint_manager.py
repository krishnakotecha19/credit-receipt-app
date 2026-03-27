"""
SharePoint Connection Manager for Dual-Entity Expense Reconciliation
=====================================================================
Uses the O365 library with client-credentials (Client ID / Secret) auth flow.

Usage:
    sp_vcare = SharePointManager(entity='vcare')
    sp_si2   = SharePointManager(entity='si2tech')

    sp_vcare.test_connection()
    sp_si2.test_connection()
"""

import os
import logging
from dotenv import load_dotenv

load_dotenv(override=True)

# ── O365 imports ────────────────────────────────────────────────────────────
try:
    from O365 import Account
except ImportError:
    raise ImportError(
        "O365 is required.  Install it with:  pip install O365"
    )

# ── Logger ──────────────────────────────────────────────────────────────────
logger = logging.getLogger("SharePointManager")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _ch = logging.StreamHandler()
    _ch.setFormatter(
        logging.Formatter("[%(levelname)s] %(name)s — %(message)s")
    )
    logger.addHandler(_ch)

# ── Entity → SharePoint library mapping ─────────────────────────────────────
_ENTITY_LIBRARY_MAP = {
    "vcare":   "Vcare_Finance",
    "si2tech": "Si2tech_Finance",
}

# Target SharePoint site (relative path under the tenant)
_SHAREPOINT_SITE_NAME = "Credit-Reciept-App2"


class SharePointManager:
    """Manages a single entity's SharePoint connection.

    Parameters
    ----------
    entity : str
        One of ``'vcare'`` or ``'si2tech'``.

    All credentials are read from the ``.env`` file:
        - ``tenantid`` — Azure AD Tenant ID
        - ``clientid`` — App Registration Client ID
        - ``secret``   — App Registration Client Secret
        - ``siteid``   — Full site ID (fallback)
    """

    def __init__(self, entity: str):
        entity = entity.strip().lower()
        if entity not in _ENTITY_LIBRARY_MAP:
            raise ValueError(
                f"Unknown entity '{entity}'. "
                f"Must be one of: {list(_ENTITY_LIBRARY_MAP.keys())}"
            )

        self.entity = entity
        self.library_name = _ENTITY_LIBRARY_MAP[entity]

        # ── Read credentials from .env ──────────────────────────────────
        self.tenant_id = os.getenv("tenantid", "")
        self.client_id = os.getenv("clientid", "")
        self.client_secret = os.getenv("secret", "")
        self.site_id = os.getenv("siteid", "")

        if not all([self.tenant_id, self.client_id, self.client_secret]):
            raise EnvironmentError(
                "Missing SharePoint credentials in .env — "
                "ensure tenantid, clientid, and secret are set."
            )

        # ── Build O365 Account (client-credentials flow) ────────────────
        credentials = (self.client_id, self.client_secret)
        self.account = Account(
            credentials,
            auth_flow_type="credentials",
            tenant_id=self.tenant_id,
        )

        # Authenticate
        self._authenticated = False
        try:
            self._authenticated = self.account.authenticate()
            if self._authenticated:
                logger.info(
                    f"[{self.entity}] Authenticated successfully."
                )
            else:
                logger.error(
                    f"[{self.entity}] Authentication returned False — "
                    "check Client ID / Secret / Tenant ID."
                )
        except Exception as exc:
            logger.error(
                f"[{self.entity}] Authentication failed: {exc}"
            )

        # Lazily populated
        self._site = None
        self._library = None

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def is_authenticated(self) -> bool:
        return self._authenticated

    # ── Internal helpers ────────────────────────────────────────────────

    def _get_site(self):
        """Connect to the SharePoint site 'Credit-Reciept-App2'."""
        if self._site is not None:
            return self._site

        if not self._authenticated:
            logger.error(
                f"[{self.entity}] Cannot get site — not authenticated."
            )
            return None

        try:
            sharepoint = self.account.sharepoint()

            # Try site-relative URL first
            site = sharepoint.get_site(
                "si2techvad.sharepoint.com",
                f"/sites/{_SHAREPOINT_SITE_NAME}",
            )
            if site:
                self._site = site
                logger.info(
                    f"[{self.entity}] Connected to site: "
                    f"{_SHAREPOINT_SITE_NAME}"
                )
                return self._site

            logger.error(
                f"[{self.entity}] Site '{_SHAREPOINT_SITE_NAME}' "
                "not found or access denied."
            )
        except Exception as exc:
            logger.error(
                f"[{self.entity}] Failed to get site "
                f"'{_SHAREPOINT_SITE_NAME}': {exc}"
            )
        return None

    def _get_library(self):
        """Get the document library for this entity."""
        if self._library is not None:
            return self._library

        site = self._get_site()
        if site is None:
            return None

        try:
            drives = site.site_storage.get_drives()
            available_names = []
            
            for drive in drives:
                available_names.append(drive.name)
                if drive.name.lower() == self.library_name.lower():
                    self._library = drive
                    logger.info(
                        f"[{self.entity}] Opened library: {self.library_name}"
                    )
                    return self._library

            logger.error(
                f"[{self.entity}] Library '{self.library_name}' not found."
            )
            print(f"[{self.entity}] ✗ Library '{self.library_name}' not found.")
            print("Available libraries on this site:")
            for name in available_names:
                print(f"  - {name}")
                
        except Exception as exc:
            logger.error(
                f"[{self.entity}] Failed to retrieve libraries: {exc}"
            )
        return None

    # ── Public API ──────────────────────────────────────────────────────

    def test_connection(self) -> bool:
        """Full connection test.

        1. Authenticates (done in __init__).
        2. Connects to the SharePoint site.
        3. Opens this entity's document library.
        4. Lists the contents of the 'Inbound Receipts' folder.

        Returns True on success, False on any failure.
        """
        print(f"\n{'='*60}")
        print(f"  SharePoint Connection Test — entity: {self.entity.upper()}")
        print(f"  Library: {self.library_name}")
        print(f"{'='*60}")

        # Step 1: Auth
        if not self._authenticated:
            print("  ✗ Authentication failed.")
            return False
        print("  ✓ Authenticated")

        # Step 2: Site
        site = self._get_site()
        if site is None:
            print(f"  ✗ Could not reach site '{_SHAREPOINT_SITE_NAME}'.")
            return False
        print(f"  ✓ Connected to site: {_SHAREPOINT_SITE_NAME}")

        # Step 3: Library
        lib = self._get_library()
        if lib is None:
            print(f"  ✗ Library '{self.library_name}' not accessible.")
            return False
        print(f"  ✓ Opened library: {self.library_name}")

        # Step 4: List "Inbound Receipts" (or "Inbound Reciepts") folder
        try:
            root_folder = lib.get_root_folder()
            inbound_folder = None

            for item in root_folder.get_child_folders():
                if item.name.lower() in ("inbound receipts", "inbound reciepts"):
                    inbound_folder = item
                    break

            if inbound_folder is None:
                print("  ✗ 'Inbound Receipts' folder not found.")
                print("    Available root folders:")
                for item in root_folder.get_child_folders():
                    print(f"      📁 {item.name}")
                return False

            print(f"  ✓ Found folder: {inbound_folder.name}")
            items = list(inbound_folder.get_items())
            if not items:
                print("    (folder is empty)")
            else:
                print(f"    Contents ({len(items)} items):")
                for item in items:
                    icon = "📁" if item.is_folder else "📄"
                    size = ""
                    if not item.is_folder and hasattr(item, "size"):
                        size = f"  ({item.size:,} bytes)"
                    print(f"      {icon} {item.name}{size}")
            return True

        except Exception as exc:
            logger.error(
                f"[{self.entity}] Error listing 'Inbound Receipts': {exc}"
            )
            print(f"  ✗ Error accessing 'Inbound Receipts': {exc}")
            return False

    def list_folder(self, folder_path: str = "/") -> list:
        """List contents of an arbitrary folder in the entity library.

        Parameters
        ----------
        folder_path : str
            Relative path inside the library, e.g. ``"/"`` or
            ``"Inbound Receipts/March 2026"``.

        Returns
        -------
        list[dict]  — ``[{"name": ..., "is_folder": ..., "size": ...}, ...]``
        """
        lib = self._get_library()
        if lib is None:
            return []

        try:
            if folder_path in ("/", ""):
                target = lib.get_root_folder()
            else:
                target = lib.get_root_folder()
                for part in folder_path.strip("/").split("/"):
                    found = None
                    for child in target.get_child_folders():
                        if child.name.lower() == part.lower():
                            found = child
                            break
                    if found is None:
                        logger.warning(
                            f"[{self.entity}] Sub-folder '{part}' not found "
                            f"in path '{folder_path}'."
                        )
                        return []
                    target = found

            results = []
            for item in target.get_items():
                results.append({
                    "name": item.name,
                    "is_folder": item.is_folder,
                    "size": getattr(item, "size", None),
                })
            return results

        except Exception as exc:
            logger.error(
                f"[{self.entity}] Error listing '{folder_path}': {exc}"
            )
            return []

    def download_file(self, file_path: str, local_dir: str = ".") -> str | None:
        """Download a single file from the entity library.

        Parameters
        ----------
        file_path : str
            Path relative to library root, e.g.
            ``"Inbound Receipts/receipt_001.jpg"``.
        local_dir : str
            Local directory to save to.

        Returns
        -------
        str | None — local file path on success, None on failure.
        """
        lib = self._get_library()
        if lib is None:
            return None

        try:
            parts = file_path.strip("/").split("/")
            folder_parts, filename = parts[:-1], parts[-1]

            target = lib.get_root_folder()
            for part in folder_parts:
                found = None
                for child in target.get_child_folders():
                    if child.name.lower() == part.lower():
                        found = child
                        break
                if found is None:
                    logger.error(
                        f"[{self.entity}] Folder '{part}' not found."
                    )
                    return None
                target = found

            for item in target.get_items():
                if not item.is_folder and item.name.lower() == filename.lower():
                    os.makedirs(local_dir, exist_ok=True)
                    local_path = os.path.join(local_dir, item.name)
                    item.download(local_dir, filename=item.name)
                    logger.info(
                        f"[{self.entity}] Downloaded: {item.name} → {local_path}"
                    )
                    return local_path

        except Exception as exc:
            logger.error(
                f"[{self.entity}] Download error: {exc}"
            )
            return None

    def sync_inbound_metadata(self) -> dict:
        """Scan Inbound Statements for PDFs and Inbound Receipts for subfolders (batches)."""
        lib = self._get_library()
        if lib is None:
            return {"statements": [], "receipt_batches": []}

        results = {"statements": [], "receipt_batches": []}
        try:
            root_folder = lib.get_root_folder()
            statements_folder = None
            receipts_folder = None
            
            for child in root_folder.get_child_folders():
                name_lower = child.name.lower()
                if "statement" in name_lower and "inbound" in name_lower:
                    statements_folder = child
                elif name_lower in ("inbound receipts", "inbound reciepts"):
                    receipts_folder = child

            if statements_folder:
                for item in statements_folder.get_items():
                    if not item.is_folder and item.name.lower().endswith('.pdf'):
                        results["statements"].append({
                            "id": item.object_id,
                            "name": item.name,
                            "size": getattr(item, "size", None)
                        })

            if receipts_folder:
                for item in receipts_folder.get_child_folders():
                    results["receipt_batches"].append({
                        "id": item.object_id,
                        "name": item.name
                    })

            return results
        except Exception as exc:
            logger.error(f"[{self.entity}] Sync metadata error: {exc}")
            return {"statements": [], "receipt_batches": []}

    def stage_files(self, statement_id: str, batch_folder_id: str, base_staging_dir: str = "static/staging") -> dict:
        """Download selected Statement PDF and all images in the selected Receipt batch.
        Skips download if local file exists and ETag matches.
        """
        lib = self._get_library()
        if lib is None:
            return {"ok": False, "error": "Library not found"}

        entity_dir = os.path.join(base_staging_dir, self.entity)
        stmt_dir = os.path.join(entity_dir, "statements")
        batch_dir_base = os.path.join(entity_dir, "batches")
        
        os.makedirs(stmt_dir, exist_ok=True)
        
        staged_statement = None
        staged_receipts = []

        def _download_with_etag(item, target_dir):
            local_path = os.path.join(target_dir, item.name)
            etag_path = local_path + ".etag"
            e_tag = getattr(item, "etag", "") # Depending on O365 version, might be eTag or etag
            if not e_tag and hasattr(item, "get_property"):
                e_tag = str(item.get_property("eTag") or "")
                
            skip = False
            if os.path.exists(local_path) and os.path.exists(etag_path):
                with open(etag_path, "r", encoding="utf-8") as f:
                    if f.read().strip() == e_tag and e_tag != "":
                        skip = True
            
            if not skip:
                item.download(target_dir, filename=item.name)
                logger.info(f"[{self.entity}] Downloaded {item.name}")
                if e_tag:
                    with open(etag_path, "w", encoding="utf-8") as f:
                        f.write(e_tag)
            else:
                logger.info(f"[{self.entity}] Skipped {item.name} (ETag match)")
                
            return local_path

        try:
            if statement_id:
                stmt_item = lib.get_item(statement_id)
                if stmt_item and not stmt_item.is_folder:
                    staged_statement = _download_with_etag(stmt_item, stmt_dir)

            if batch_folder_id:
                batch_folder = lib.get_item(batch_folder_id)
                if batch_folder and batch_folder.is_folder:
                    batch_dir = os.path.join(batch_dir_base, batch_folder.name)
                    os.makedirs(batch_dir, exist_ok=True)
                    
                    for item in batch_folder.get_items():
                        if not item.is_folder:
                            ext = item.name.lower().split('.')[-1]
                            if ext in ('png', 'jpg', 'jpeg', 'pdf'):
                                local_path = _download_with_etag(item, batch_dir)
                                staged_receipts.append(local_path)

            return {
                "ok": True, 
                "statement": staged_statement, 
                "receipts": staged_receipts
            }
        except Exception as exc:
            logger.error(f"[{self.entity}] Staging error: {exc}")
            return {"ok": False, "error": str(exc)}


# ── Quick self-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing Vcare connection...")
    sp_vcare = SharePointManager(entity="vcare")
    sp_vcare.test_connection()

    print("\nTesting Si2tech connection...")
    sp_si2 = SharePointManager(entity="si2tech")
    sp_si2.test_connection()
