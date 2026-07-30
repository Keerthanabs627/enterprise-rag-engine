async function fetchDocuments() {
    try {
        const res = await fetch('/api/documents');
        const data = await res.json();
        const list = document.getElementById('documentList');
        if (data.documents.length === 0) {
            list.innerHTML = '<li><em>No documents uploaded yet.</em></li>';
            return;
        }
        list.innerHTML = data.documents.map(d => 
            `<li><strong>${d.title}</strong> - <em>${d.status}</em> (${d.chunks_count} chunks)</li>`
        ).join('');
    } catch (err) {
        console.error("Error fetching documents:", err);
    }
}

async function uploadFile() {
    const fileInput = document.getElementById('fileInput');
    if (!fileInput.files[0]) return alert("Please select a file first!");

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    document.getElementById('uploadStatus').innerText = "Uploading to queue...";
    try {
        await fetch('/api/documents/upload', { method: 'POST', body: formData });
        document.getElementById('uploadStatus').innerText = "Uploaded successfully. Processing in background...";
        fileInput.value = "";
        setTimeout(fetchDocuments, 1500);
    } catch (err) {
        document.getElementById('uploadStatus').innerText = "Upload failed. Try again.";
    }
}

async function submitQuery() {
    const queryInput = document.getElementById('queryInput');
    const query = queryInput.value.trim();
    if (!query) return;

    const sourcesBox = document.getElementById('sourcesContainer');
    const responseBox = document.getElementById('responseContainer');
    
    sourcesBox.innerHTML = "Retrieving relevant document chunks...";
    responseBox.innerText = "";

    try {
        const response = await fetch('/api/query/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query })
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let buffer = "";

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n\n");
            
            buffer = lines.pop();

            for (const chunk of lines) {
                if (chunk.startsWith("event: sources")) {
                    const rawData = chunk.replace("event: sources\ndata: ", "");
                    try {
                        const sources = JSON.parse(rawData);
                        if (sources.length === 0) {
                            sourcesBox.innerHTML = "<em>No matching sources found above threshold.</em>";
                        } else {
                            sourcesBox.innerHTML = sources.map(s => 
                                `<div><strong>[${s.document}]</strong> (Score: ${s.score})<br><small>"${s.text}"</small></div><br>`
                            ).join('');
                        }
                    } catch (e) {
                        sourcesBox.innerHTML = "<em>Error parsing sources.</em>";
                    }
                } else if (chunk.startsWith("data: ")) {
                    const token = chunk.replace("data: ", "");
                    if (token === "[DONE]") {
                        break;
                    }
                    responseBox.innerText += token + " ";
                }
            }
        }
    } catch (err) {
        responseBox.innerText = "Error executing streaming query.";
    }

    fetchDocuments();
}

fetchDocuments();