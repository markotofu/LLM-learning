/* ML Atlas – legal texts (terms of use, privacy policy) and the copyright notice.
 * To change the owner, contact link or dates, edit the fields below; the texts use {owner}, {year}, {contact} and {updated}. */
(function (global) {
  const MLA = (global.MLA = global.MLA || {});
  MLA.data = MLA.data || {};

  const legal = {
    owner: 'markotofu',                          // copyright holder shown in the app and in the legal texts
    year: '2026',
    updated: '20 November 2026',
    contact: 'https://github.com/markotofu',     // where people can reach the owner
    codeLicense: 'MIT License',
    terms: `
*Last updated: {updated}*

These terms apply to ML Atlas ("ML Atlas", "the app"), a free educational web app made by {owner} ("we", "us"). By using the app you agree to them. If you do not agree, please do not use the app.

### 1. What ML Atlas is

ML Atlas is a free learning resource about machine learning, neural networks and large language models: lessons, runnable code samples, quizzes, flashcards and practice problems. It has no accounts and no paid features. It runs entirely in your web browser.

### 2. Educational use only — no professional advice

The content is for learning. It is not professional, legal, medical, financial or security advice, and it may contain mistakes or become out of date. Check important information against authoritative sources before you rely on it, and test code before using it in real systems.

### 3. Your progress and your code

Your progress, notes, settings and the code you write are stored **in your own browser**, not on our servers (see the [Privacy Policy](#/privacy)). You are responsible for keeping backups — use **Progress & Sync → Export** — because clearing your browser data or switching browsers or devices can delete them. You keep all rights to the code and notes you write.

### 4. Running code

Python code runs inside your browser in an isolated worker. You are responsible for the code you write and run, including any network requests it makes. Do not use the app to attack, overload or gain unauthorized access to any system or data.

### 5. Acceptable use

Do not misuse the app: do not try to disrupt or overload it or its hosting, do not use it to break the law or infringe anyone else's rights, and do not remove or alter copyright notices.

### 6. Copyright and licences

The lessons, quizzes, practice problems, explanations, glossary, flashcards and other educational content of ML Atlas are © {year} {owner}. All rights reserved. You may use them for your own personal, non-commercial learning, and you may quote short passages with attribution. Do not republish, sell or redistribute the content, or use it to build a competing product, without written permission.

The app's source code (its JavaScript, HTML and CSS) is released under the {codeLicense}; see the LICENSE file that accompanies it. Third-party software included in or loaded by the app — such as Vue and Pyodide with its Python packages — remains under its own licence; the [third-party notices](THIRD_PARTY_NOTICES.md) list each component and its licence. Names and trademarks of products, companies and projects mentioned in the lessons belong to their owners and are used only to identify them. Referenced papers and websites remain the work of their authors.

### 7. Third-party services and links

The app is delivered by its hosting provider (for example GitHub Pages), and the in-browser Python runtime is downloaded from a third-party content delivery network (jsDelivr). Lessons link to external websites. These services and sites are not under our control and have their own terms and privacy policies.

### 8. No warranty

The app and its content are provided **"as is" and "as available"**, without warranties of any kind, express or implied, including fitness for a particular purpose, accuracy, availability and non-infringement.

### 9. Limitation of liability

To the fullest extent permitted by law, we are not liable for any indirect, incidental or consequential damages, or for any loss of data, progress, profits or opportunities, arising from your use of — or inability to use — the app or its content. Nothing in these terms limits liability that cannot be limited under the law that applies to you.

### 10. Changes

We may update these terms. The date at the top shows the latest version; continuing to use the app after a change means you accept the updated terms.

### 11. Reporting errors and copyright concerns

If you find a mistake in the content, or believe something in ML Atlas infringes your copyright or other rights, please tell us through the contact below, with enough detail to identify the material. We will review reports promptly and correct or remove material where appropriate.

### 12. Contact

Questions about these terms: [{contact}]({contact}).
`,
    privacy: `
*Last updated: {updated}*

This policy explains what happens to your information when you use ML Atlas, a free educational web app made by {owner} ("we", "us").

### In short

- **No accounts, no sign-up.** We never ask for your name, email address or password.
- **Your data stays on your device.** Progress, code, notes and settings are saved in your browser's local storage. We do not have a server that receives them.
- **No cookies, analytics, ads or tracking** are used by ML Atlas.
- Like every website, the services that **deliver the app's files** — the web host and the content delivery network for the Python runtime — receive standard technical information such as your IP address.

### 1. Information stored on your device

ML Atlas saves the following in your browser so that it works across visits. It never leaves your device unless you export it yourself.

- **Your learning data** (local storage key \`mla.state.v1\`): concepts read, quiz and problem results, the code you write in problems and the playground, notes, flashcard schedules, XP, streaks, achievements, settings (theme, editor size, learning goal) and an optional display name (default "Learner").
- **A technical preference** (local storage key \`mla.pyMirror\`): which download server for the Python runtime responded fastest.
- **Cached app files**: the app's own files are kept in your browser's cache storage so it works offline, and the Python runtime is kept in the browser's normal download cache.

You can delete all of this at any time with **Progress & Sync → Danger zone → Erase all progress**, or by clearing this site's data in your browser settings.

### 2. Export, import and sync codes

When you export your progress, the app creates a file or a "sync code" **on your device**. It contains your learning data and code. You decide where it goes — treat it like any other personal file. Importing reads such a file or code locally; nothing is uploaded.

### 3. Services that deliver the app

- **Hosting.** The app's files are served by its web host (for example GitHub Pages). When your browser requests them, the host receives standard request information such as your IP address, browser type and the time of the request, and may keep logs for security and operation. See the [GitHub General Privacy Statement](https://docs.github.com/en/site-policy/privacy-policies/github-general-privacy-statement). If you run ML Atlas from your own computer or another server, that server's policies apply instead.
- **Python runtime.** The first time you run code, your browser downloads the Pyodide Python runtime and Python packages from the jsDelivr content delivery network (cdn.jsdelivr.net or one of its alternative hosts). jsDelivr receives the same kind of request information. See the [jsDelivr privacy policy](https://www.jsdelivr.com/terms/privacy-policy-jsdelivr-net).
- **Links.** If you follow a link from a lesson to another website, that website's own privacy policy applies.

### 4. Cookies and tracking

ML Atlas does not set cookies and does not use analytics, advertising, fingerprinting or other tracking technologies.

### 5. Code you run and API keys

Code runs inside your browser. If code you write makes network requests, those requests go from your browser to the address you chose. ML Atlas never asks for API keys or passwords; examples that call external APIs (such as LLM APIs) are meant to be run on your own computer, with your own keys, under that provider's terms.

### 6. Children

ML Atlas does not knowingly collect personal information from anyone, including children, because it does not collect personal information at all.

### 7. Your rights

Because we do not receive or store your personal data, there is nothing for us to access, correct or delete on our side — you control everything stored by ML Atlas through your browser. For data held by the hosting or delivery services, use the contact options in their policies.

### 8. Changes

We may update this policy. The date at the top shows the latest version.

### 9. Contact

Questions about privacy: [{contact}]({contact}).
`
  };

  legal.render = function (text) {
    return String(text)
      .replace(/\{owner\}/g, legal.owner)
      .replace(/\{year\}/g, legal.year)
      .replace(/\{updated\}/g, legal.updated)
      .replace(/\{contact\}/g, legal.contact)
      .replace(/\{codeLicense\}/g, legal.codeLicense);
  };
  legal.copyright = function () { return '© ' + legal.year + ' ' + legal.owner; };

  MLA.data.legal = legal;
})(typeof globalThis !== 'undefined' ? globalThis : window);
