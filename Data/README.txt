Dataset Description
---------------------------------------

This dataset contains three folds of documents: privacy policy, terms of service, and miscellaneous. Each document comes with a gold standard file ('gold.html') that contains the label (e.g., whether a piece of text is a section title or a prose paragraph) of each text segment in the original document. "h2" tags represent section titles, "p" tags represent prose paragraph, and "div" tags represent miscellany class. "TOS.html", "priv.html", and "Misc.html" are the original HTML documents of terms of service, privacy policy, and miscellaneous topics.

Each fold of documents contains a 'note.txt' file that documents the edit history within that fold. Each fold also has a csv file that records the time used to manually label each document.

Due to the space constraints of submission site, we deleted the static files that support each HTML file (e.g., javascript and CSS files). As a result, if you open an HTML document through a browser, its visual appearance may get changed, but model training will not be affected.