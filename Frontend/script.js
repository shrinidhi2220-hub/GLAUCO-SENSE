/* =========================================================
   GLAUCO-SENSE
   Frontend application logic

   Handles:
   - Navigation
   - Mobile menu
   - Fundus image selection
   - Image preview
   - Real FastAPI prediction
   - Screening result display
========================================================= */

document.addEventListener("DOMContentLoaded", () => {

    /* =====================================================
       CONFIGURATION
    ===================================================== */

    const API_BASE_URL = "http://localhost:8000";

    const PREDICT_ENDPOINT =
        `${API_BASE_URL}/api/predict`;


    /* =====================================================
       GLOBAL STATE
    ===================================================== */

    const state = {
        selectedFile: null,
        previewUrl: null,
        lastResult: null,
        analyzing: false,
    };


    /* =====================================================
       ELEMENT REFERENCES
    ===================================================== */

    const sidebar =
        document.getElementById("sidebar");

    const mobileMenuButton =
        document.getElementById("mobileMenuButton");

    const navButtons =
        document.querySelectorAll(".nav-button");

    const pageSections =
        document.querySelectorAll(".page-section");


    /* =====================================================
       NAVIGATION
    ===================================================== */

    function navigateTo(sectionId) {

        const target =
            document.getElementById(sectionId);

        if (!target) {
            console.warn(
                "Section not found:",
                sectionId
            );
            return;
        }

        pageSections.forEach(section => {
            section.classList.remove(
                "active-section"
            );
        });

        target.classList.add(
            "active-section"
        );

        navButtons.forEach(button => {

            button.classList.remove(
                "active"
            );

            if (
                button.dataset.section ===
                sectionId
            ) {
                button.classList.add(
                    "active"
                );
            }
        });

        if (
            window.innerWidth <= 700 &&
            sidebar
        ) {
            sidebar.classList.remove(
                "menu-open"
            );
        }

        window.scrollTo({
            top: 0,
            behavior: "smooth",
        });
    }


    navButtons.forEach(button => {

        button.addEventListener(
            "click",
            () => {
                navigateTo(
                    button.dataset.section
                );
            }
        );

    });


    /* =====================================================
       MOBILE MENU
    ===================================================== */

    if (mobileMenuButton && sidebar) {

        mobileMenuButton.addEventListener(
            "click",
            () => {
                sidebar.classList.toggle(
                    "menu-open"
                );
            }
        );

    }


    /* =====================================================
       HELPER
    ===================================================== */

    function setText(id, value) {

        const element =
            document.getElementById(id);

        if (element) {
            element.textContent = value;
        }
    }


    function setButtonState(
        button,
        disabled,
        text
    ) {

        if (!button) {
            return;
        }

        button.disabled = disabled;

        if (text !== undefined) {
            button.textContent = text;
        }
    }


    /* =====================================================
       FUNDUS IMAGE ELEMENTS
    ===================================================== */

    const fundusInput =
        document.getElementById(
            "fundusInput"
        );

    const fundusPreview =
        document.getElementById(
            "fundusPreview"
        );

    const imagePreviewPlaceholder =
        document.getElementById(
            "imagePreviewPlaceholder"
        );

    const selectedFileName =
        document.getElementById(
            "selectedFileName"
        );

    const analyzeButton =
        document.getElementById(
            "analyzeButton"
        );

    const analysisStatus =
        document.getElementById(
            "analysisStatus"
        );


    /* =====================================================
       RESULT ELEMENTS
    ===================================================== */

    const resultTitle =
        document.getElementById(
            "resultTitle"
        );

    const resultDescription =
        document.getElementById(
            "resultDescription"
        );

    const resultClass =
        document.getElementById(
            "resultClass"
        );

    const resultConfidence =
        document.getElementById(
            "resultConfidence"
        );

    const resultClassIndex =
        document.getElementById(
            "resultClassIndex"
        );

    const gonPositiveProbability =
        document.getElementById(
            "gonPositiveProbability"
        );

    const gonNegativeProbability =
        document.getElementById(
            "gonNegativeProbability"
        );

    const resultNote =
        document.getElementById(
            "resultNote"
        );


    /* =====================================================
       IMAGE PREVIEW
    ===================================================== */

    function clearPreview() {

        if (state.previewUrl) {
            URL.revokeObjectURL(
                state.previewUrl
            );

            state.previewUrl = null;
        }

        if (fundusPreview) {
            fundusPreview.style.display =
                "none";

            fundusPreview.removeAttribute(
                "src"
            );
        }

        if (imagePreviewPlaceholder) {
            imagePreviewPlaceholder.style.display =
                "inline";
        }
    }


    function showPreview(file) {

        clearPreview();

        state.previewUrl =
            URL.createObjectURL(file);

        if (fundusPreview) {

            fundusPreview.src =
                state.previewUrl;

            fundusPreview.style.display =
                "block";
        }

        if (imagePreviewPlaceholder) {
            imagePreviewPlaceholder.style.display =
                "none";
        }
    }


    /* =====================================================
       RESET RESULT
    ===================================================== */

    function resetResult() {

        state.lastResult = null;

        setText(
            "resultTitle",
            "AWAITING ANALYSIS"
        );

        setText(
            "resultDescription",
            "Upload and analyze a fundus image to generate a model result."
        );

        setText(
            "resultClass",
            "—"
        );

        setText(
            "resultConfidence",
            "—"
        );

        setText(
            "resultClassIndex",
            "—"
        );

        setText(
            "gonPositiveProbability",
            "—"
        );

        setText(
            "gonNegativeProbability",
            "—"
        );

        setText(
            "resultNote",
            "Research-model output will appear here after analysis."
        );
    }


    /* =====================================================
       FILE SELECTION
    ===================================================== */

    if (fundusInput) {

        fundusInput.addEventListener(
            "change",
            event => {

                const files =
                    event.target.files;

                if (
                    !files ||
                    files.length === 0
                ) {

                    state.selectedFile =
                        null;

                    if (selectedFileName) {
                        selectedFileName.textContent =
                            "No image selected.";
                    }

                    clearPreview();
                    resetResult();

                    setButtonState(
                        analyzeButton,
                        true,
                        "🔍 ANALYZE FUNDUS IMAGE"
                    );

                    if (analysisStatus) {
                        analysisStatus.textContent =
                            "Waiting for an image.";
                    }

                    return;
                }

                const file = files[0];

                if (
                    !file.type ||
                    !file.type.startsWith(
                        "image/"
                    )
                ) {

                    state.selectedFile =
                        null;

                    if (selectedFileName) {
                        selectedFileName.textContent =
                            "Please select a valid image file.";
                    }

                    clearPreview();
                    resetResult();

                    setButtonState(
                        analyzeButton,
                        true,
                        "🔍 ANALYZE FUNDUS IMAGE"
                    );

                    if (analysisStatus) {
                        analysisStatus.textContent =
                            "Invalid file type.";
                    }

                    return;
                }

                state.selectedFile =
                    file;

                if (selectedFileName) {
                    selectedFileName.textContent =
                        `${file.name} • ${formatBytes(file.size)}`;
                }

                showPreview(file);
                resetResult();

                setButtonState(
                    analyzeButton,
                    false,
                    "🔍 ANALYZE FUNDUS IMAGE"
                );

                if (analysisStatus) {
                    analysisStatus.textContent =
                        "Image ready for analysis.";
                }
            }
        );
    }


    /* =====================================================
       FILE SIZE FORMAT
    ===================================================== */

    function formatBytes(bytes) {

        if (!Number.isFinite(bytes)) {
            return "Unknown size";
        }

        if (bytes < 1024) {
            return `${bytes} B`;
        }

        if (bytes < 1024 * 1024) {
            return `${(
                bytes / 1024
            ).toFixed(1)} KB`;
        }

        return `${(
            bytes /
            (1024 * 1024)
        ).toFixed(2)} MB`;
    }


    /* =====================================================
       RESULT RENDERING
    ===================================================== */

    function renderResult(result) {

        state.lastResult =
            result;

        const prediction =
            result.prediction;

        const confidence =
            Number(result.confidence);

        const classIndex =
            result.class_index;

        const probabilities =
            result.probabilities || {};

        const gonPositive =
            Number(
                probabilities["GON+"] ?? 0
            );

        const gonNegative =
            Number(
                probabilities["GON-"] ?? 0
            );

        setText(
            "resultTitle",
            prediction
        );

        setText(
            "resultDescription",
            prediction === "GON+"
                ? "The research model classified the uploaded image as GON+."
                : "The research model classified the uploaded image as GON-."
        );

        setText(
            "resultClass",
            prediction
        );

        setText(
            "resultConfidence",
            Number.isFinite(confidence)
                ? `${(
                    confidence * 100
                ).toFixed(2)}%`
                : "—"
        );

        setText(
            "resultClassIndex",
            classIndex ?? "—"
        );

        setText(
            "gonPositiveProbability",
            `${(
                gonPositive * 100
            ).toFixed(2)}%`
        );

        setText(
            "gonNegativeProbability",
            `${(
                gonNegative * 100
            ).toFixed(2)}%`
        );

        setText(
            "resultNote",
            result.note ||
            "Research-model output only."
        );

        navigateTo("result");
    }


    /* =====================================================
       API ERROR MESSAGE
    ===================================================== */

    async function extractApiError(
        response
    ) {

        try {

            const body =
                await response.json();

            if (
                body &&
                typeof body.detail ===
                "string"
            ) {
                return body.detail;
            }

            if (
                body &&
                typeof body.detail !==
                "undefined"
            ) {
                return JSON.stringify(
                    body.detail
                );
            }

            return JSON.stringify(
                body
            );

        } catch {

            return (
                `Request failed with status ` +
                `${response.status}.`
            );
        }
    }


    /* =====================================================
       API PREDICTION
    ===================================================== */

    async function analyzeFundus() {

        if (!state.selectedFile) {

            if (analysisStatus) {
                analysisStatus.textContent =
                    "Please select an image first.";
            }

            return;
        }

        if (state.analyzing) {
            return;
        }

        state.analyzing = true;

        setButtonState(
            analyzeButton,
            true,
            "⏳ ANALYZING..."
        );

        if (analysisStatus) {
            analysisStatus.textContent =
                "Sending image to the GLAUCO-SENSE research model...";
        }

        try {

            const formData =
                new FormData();

            formData.append(
                "file",
                state.selectedFile
            );

            const response =
                await fetch(
                    PREDICT_ENDPOINT,
                    {
                        method: "POST",
                        body: formData,
                    }
                );

            if (!response.ok) {

                const message =
                    await extractApiError(
                        response
                    );

                throw new Error(
                    message
                );
            }

            const result =
                await response.json();

            renderResult(result);

            if (analysisStatus) {
                analysisStatus.textContent =
                    "Analysis completed successfully.";
            }

        } catch (error) {

            console.error(
                "GLAUCO-SENSE prediction failed:",
                error
            );

            resetResult();

            setText(
                "resultTitle",
                "ANALYSIS FAILED"
            );

            setText(
                "resultDescription",
                error?.message ||
                "Unable to connect to the research model."
            );

            setText(
                "resultNote",
                "Check that the FastAPI backend is running on port 8000."
            );

            navigateTo("result");

            if (analysisStatus) {
                analysisStatus.textContent =
                    "Analysis failed.";
            }

        } finally {

            state.analyzing = false;

            setButtonState(
                analyzeButton,
                !state.selectedFile,
                "🔍 ANALYZE FUNDUS IMAGE"
            );
        }
    }


    if (analyzeButton) {

        analyzeButton.addEventListener(
            "click",
            analyzeFundus
        );

    }


    /* =====================================================
       API HEALTH CHECK
    ===================================================== */

    async function checkBackendHealth() {

        try {

            const response =
                await fetch(
                    `${API_BASE_URL}/health`,
                    {
                        method: "GET",
                    }
                );

            if (!response.ok) {
                throw new Error(
                    "Backend health check failed."
                );
            }

            const data =
                await response.json();

            console.log(
                "GLAUCO-SENSE backend:",
                data
            );

        } catch (error) {

            console.warn(
                "Backend is not currently reachable:",
                error.message
            );

        }
    }


    /* =====================================================
       INITIALIZATION
    ===================================================== */

    function initializeApp() {

        navigateTo(
            "overview"
        );

        resetResult();

        if (analysisStatus) {
            analysisStatus.textContent =
                "Waiting for an image.";
        }

        checkBackendHealth();
    }


    initializeApp();


    /* =====================================================
       CLEANUP
    ===================================================== */

    window.addEventListener(
        "beforeunload",
        () => {

            if (state.previewUrl) {

                URL.revokeObjectURL(
                    state.previewUrl
                );

                state.previewUrl = null;
            }

        }
    );

});