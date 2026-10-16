#pragma once

// A made-up help centre for a made-up notes app. Short articles, the kind of text a
// search box in a real application sits on top of.

#include <array>
#include <string_view>

struct Article {
    std::string_view category;
    std::string_view title;
    std::string_view body;
};

inline constexpr std::array kArticles = {
    Article{"Account", "Reset your password",
            "Use the link on the sign-in page to get a reset email, then choose a new password."},
    Article{"Account", "Turn on two-factor authentication",
            "Add an authenticator app so signing in needs a code as well as your password."},
    Article{"Account", "Change the email address on your account",
            "Update the address you sign in with and where receipts are sent."},
    Article{"Account", "Delete your account",
            "Permanently remove your account and every note stored in it."},
    Article{"Billing", "Cancel your subscription",
            "Stop the next renewal. You keep paid features until the end of the period."},
    Article{"Billing", "Request a refund",
            "Payments made in the last 14 days can be returned to the original card."},
    Article{"Billing", "Update your payment card",
            "Replace an expired or lost card before the next renewal fails."},
    Article{"Billing", "Switch between monthly and yearly plans",
            "Yearly billing costs less per month. The change applies at the next renewal."},
    Article{"Billing", "Download invoices for your company",
            "Get PDF receipts with your company name and VAT number for accounting."},
    Article{"Sync", "Notes are not syncing between devices",
            "Check that every device is signed in to the same account and online."},
    Article{"Sync", "Work offline",
            "Notes edited without a connection are saved locally and upload when you reconnect."},
    Article{"Sync", "Resolve a conflicted copy",
            "When two devices edit the same note offline, both versions are kept side by side."},
    Article{"Notes", "Recover a deleted note",
            "Deleted notes stay in the trash for 30 days before they are removed for good."},
    Article{"Notes", "See earlier versions of a note",
            "Version history lets you view and restore any saved state of a note."},
    Article{"Notes", "Organise notes with tags",
            "Add tags to group related notes and filter the sidebar by them."},
    Article{"Notes", "Pin a note to the top",
            "Keep an important note above everything else in its notebook."},
    Article{"Sharing", "Share a note with someone",
            "Send a link that lets another person read or edit a single note."},
    Article{"Sharing", "Stop sharing a note",
            "Revoke a link so people who had it can no longer open the note."},
    Article{"Sharing", "Collaborate in real time",
            "Several people can edit a shared note at once and see each other's cursors."},
    Article{"Import and export", "Export all your notes",
            "Download everything as Markdown files in a single zip archive."},
    Article{"Import and export", "Import notes from another app",
            "Bring in notes exported from other apps as Markdown, text or HTML."},
    Article{"Import and export", "Print a note",
            "Print a note or save it as a PDF with its formatting intact."},
    Article{"Privacy", "Encrypt a notebook",
            "Protect a notebook with its own passphrase so only you can read it."},
    Article{"Privacy", "Where your data is stored",
            "Notes are stored in data centres in the EU and encrypted at rest."},
};
