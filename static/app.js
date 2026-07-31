// ============================================================
// ENTERPRISE RAG ENGINE - FRONTEND
// ============================================================


// ============================================================
// 1. LOAD DOCUMENTS
// ============================================================

async function fetchDocuments() {

    const list = document.getElementById("documentList");

    try {

        const response = await fetch("/api/documents");

        if (!response.ok) {
            throw new Error(
                `HTTP ${response.status}`
            );
        }

        const data = await response.json();

        const documents = data.documents || [];

        if (documents.length === 0) {

            list.innerHTML =
                "<li><em>No documents uploaded yet.</em></li>";

            return;
        }

        list.innerHTML = documents
            .map(document => {

                const title = escapeHTML(
                    document.title || "Unknown document"
                );

                const status = escapeHTML(
                    document.status || "unknown"
                );

                const chunks =
                    document.chunks_count ?? 0;

                return `
                    <li>
                        <strong>${title}</strong>
                        -
                        <em>${status}</em>
                        (${chunks} chunks)
                    </li>
                `;
            })
            .join("");

    } catch (error) {

        console.error(
            "Error fetching documents:",
            error
        );

        list.innerHTML =
            "<li><em>Unable to load documents.</em></li>";
    }
}


// ============================================================
// 2. UPLOAD DOCUMENT
// ============================================================

async function uploadFile() {

    const fileInput =
        document.getElementById("fileInput");

    const statusBox =
        document.getElementById("uploadStatus");

    const file =
        fileInput.files[0];

    if (!file) {

        alert(
            "Please select a file first!"
        );

        return;
    }


    const formData =
        new FormData();

    formData.append(
        "file",
        file
    );


    statusBox.innerText =
        "Uploading document...";


    try {

        const response = await fetch(
            "/api/documents/upload",
            {
                method: "POST",
                body: formData
            }
        );


        if (!response.ok) {

            const errorText =
                await response.text();

            throw new Error(
                errorText
            );
        }


        const data =
            await response.json();


        statusBox.innerText =
            "Uploaded successfully. Processing in background...";


        console.log(
            "Upload response:",
            data
        );


        fileInput.value = "";


        // Refresh immediately
        await fetchDocuments();


        // Refresh again while background
        // processing is likely running
        setTimeout(
            fetchDocuments,
            2000
        );

        setTimeout(
            fetchDocuments,
            5000
        );


    } catch (error) {

        console.error(
            "Upload error:",
            error
        );

        statusBox.innerText =
            "Upload failed. Please try again.";
    }
}


// ============================================================
// 3. SUBMIT RAG QUERY
// ============================================================

async function submitQuery() {

    const queryInput =
        document.getElementById("queryInput");

    const sourcesBox =
        document.getElementById("sourcesContainer");

    const responseBox =
        document.getElementById("responseContainer");


    const query =
        queryInput.value.trim();


    if (!query) {

        alert(
            "Please enter a question."
        );

        return;
    }


    // Reset previous result

    sourcesBox.innerHTML =
        "<em>Searching knowledge base...</em>";

    responseBox.innerHTML =
        "<em>Generating answer...</em>";


    try {

        // ====================================================
        // CALL STREAMING API
        // ====================================================

        const response = await fetch(
            "/api/query/stream",
            {
                method: "POST",

                headers: {
                    "Content-Type":
                        "application/json"
                },

                body: JSON.stringify({
                    query: query
                })
            }
        );


        if (!response.ok) {

            const errorText =
                await response.text();

            throw new Error(
                `Query failed: ${response.status} ${errorText}`
            );
        }


        if (!response.body) {

            throw new Error(
                "Streaming response body is unavailable."
            );
        }


        // Clear loading message once
        // connection succeeds

        responseBox.innerHTML = "";


        // ====================================================
        // READ STREAM
        // ====================================================

        const reader =
            response.body.getReader();


        const decoder =
            new TextDecoder("utf-8");


        let buffer = "";

        let answerText = "";

        let receivedSources = false;

        let streamFinished = false;


        while (!streamFinished) {

            const {
                done,
                value
            } = await reader.read();


            if (done) {

                // Process any remaining text
                buffer += decoder.decode();

                break;
            }


            buffer += decoder.decode(
                value,
                {
                    stream: true
                }
            );


            // Normalize Windows line endings
            buffer =
                buffer.replace(
                    /\r\n/g,
                    "\n"
                );


            // SSE events are separated
            // by a blank line

            const events =
                buffer.split("\n\n");


            // Keep unfinished event
            buffer =
                events.pop() || "";


            for (const eventBlock of events) {

                if (!eventBlock.trim()) {
                    continue;
                }


                const parsed =
                    parseSSEEvent(
                        eventBlock
                    );


                // ============================================
                // SOURCES EVENT
                // ============================================

                if (
                    parsed.event ===
                    "sources"
                ) {

                    receivedSources = true;

                    try {

                        const sources =
                            JSON.parse(
                                parsed.data
                            );


                        renderSources(
                            sourcesBox,
                            sources
                        );


                    } catch (error) {

                        console.error(
                            "Source JSON parsing error:",
                            error,
                            parsed.data
                        );


                        sourcesBox.innerHTML =
                            "<em>Unable to display retrieved sources.</em>";
                    }

                    continue;
                }


                // ============================================
                // END EVENT
                // ============================================

                if (
                    parsed.event ===
                    "end"
                ) {

                    streamFinished = true;

                    continue;
                }


                // ============================================
                // ANSWER TOKEN
                // ============================================

                if (parsed.data) {

                    if (
                        parsed.data.trim() ===
                        "[DONE]"
                    ) {

                        streamFinished = true;

                        continue;
                    }


                    answerText +=
                        parsed.data;


                    renderAnswer(
                        responseBox,
                        answerText
                    );
                }
            }
        }


        // ====================================================
        // HANDLE REMAINING BUFFER
        // ====================================================

        if (buffer.trim()) {

            const parsed =
                parseSSEEvent(
                    buffer
                );


            if (
                parsed.event ===
                "sources"
            ) {

                try {

                    const sources =
                        JSON.parse(
                            parsed.data
                        );


                    renderSources(
                        sourcesBox,
                        sources
                    );


                    receivedSources = true;


                } catch (error) {

                    console.error(
                        error
                    );
                }
            }


            else if (
                parsed.data &&
                parsed.data.trim() !==
                    "[DONE]"
            ) {

                answerText +=
                    parsed.data;


                renderAnswer(
                    responseBox,
                    answerText
                );
            }
        }


        // ====================================================
        // EMPTY ANSWER FALLBACK
        // ====================================================

        if (
            answerText.trim() === ""
        ) {

            responseBox.innerHTML =
                "<em>No answer was generated.</em>";
        }


        if (!receivedSources) {

            sourcesBox.innerHTML =
                "<em>No retrieval sources were returned.</em>";
        }


    } catch (error) {

        console.error(
            "Streaming query error:",
            error
        );


        responseBox.innerHTML =
            `
            <strong>Unable to generate answer.</strong>
            <br>
            <small>
                ${escapeHTML(
                    error.message
                )}
            </small>
            `;

    }


    // Refresh document status
    fetchDocuments();
}


// ============================================================
// 4. PARSE SSE EVENT
// ============================================================

function parseSSEEvent(
    eventBlock
) {

    const lines =
        eventBlock.split("\n");


    let eventName =
        "message";


    const dataLines =
        [];


    for (const line of lines) {

        if (
            line.startsWith(
                "event:"
            )
        ) {

            eventName =
                line
                    .slice(6)
                    .trim();

        }

        else if (
            line.startsWith(
                "data:"
            )
        ) {

            let data =
                line.slice(5);


            if (
                data.startsWith(" ")
            ) {

                data =
                    data.slice(1);
            }


            dataLines.push(
                data
            );
        }
    }


    return {

        event:
            eventName,

        data:
            dataLines.join("\n")
    };
}


// ============================================================
// 5. RENDER RETRIEVED SOURCES
// ============================================================

function renderSources(
    container,
    sources
) {

    if (
        !Array.isArray(sources) ||
        sources.length === 0
    ) {

        container.innerHTML =
            `
            <em>
                No matching sources found
                above the retrieval threshold.
            </em>
            `;

        return;
    }


    container.innerHTML =
        sources
            .map(
                (
                    source,
                    index
                ) => {

                    const document =
                        escapeHTML(
                            source.document ||
                            "Unknown document"
                        );


                    const text =
                        escapeHTML(
                            source.text || ""
                        );


                    const score =
                        typeof source.score ===
                        "number"

                            ? source.score.toFixed(4)

                            : escapeHTML(
                                String(
                                    source.score ??
                                    "N/A"
                                )
                            );


                    const page =
                        source.page !== null &&
                        source.page !== undefined

                            ? `Page ${escapeHTML(
                                String(
                                    source.page
                                )
                            )}`

                            : "Page unavailable";


                    return `
                        <div class="source-card">

                            <div class="source-header">

                                <strong>
                                    Source ${index + 1}
                                </strong>

                            </div>


                            <div>

                                <strong>
                                    ${document}
                                </strong>

                            </div>


                            <div>

                                ${page}
                                &nbsp;•&nbsp;
                                Similarity:
                                ${score}

                            </div>


                            <div class="source-text">

                                ${text}

                            </div>

                        </div>
                    `;
                }
            )
            .join("");
}


// ============================================================
// 6. RENDER ANSWER
// ============================================================

function renderAnswer(
    container,
    text
) {

    // Escape HTML first so model output
    // cannot inject HTML into the page.

    let safe =
        escapeHTML(text);


    // Basic Markdown support

    safe =
        safe.replace(
            /\*\*(.*?)\*\*/g,
            "<strong>$1</strong>"
        );


    safe =
        safe.replace(
            /\*(.*?)\*/g,
            "<em>$1</em>"
        );


    safe =
        safe.replace(
            /\n/g,
            "<br>"
        );


    container.innerHTML =
        safe;
}


// ============================================================
// 7. ESCAPE HTML
// ============================================================

function escapeHTML(
    value
) {

    return String(value)

        .replace(
            /&/g,
            "&amp;"
        )

        .replace(
            /</g,
            "&lt;"
        )

        .replace(
            />/g,
            "&gt;"
        )

        .replace(
            /"/g,
            "&quot;"
        )

        .replace(
            /'/g,
            "&#039;"
        );
}


// ============================================================
// 8. ENTER KEY SUPPORT
// ============================================================

document.addEventListener(
    "DOMContentLoaded",
    () => {

        const queryInput =
            document.getElementById(
                "queryInput"
            );


        if (queryInput) {

            queryInput.addEventListener(
                "keydown",
                event => {

                    if (
                        event.key ===
                        "Enter"
                    ) {

                        event.preventDefault();

                        submitQuery();
                    }
                }
            );
        }


        fetchDocuments();
    }
);