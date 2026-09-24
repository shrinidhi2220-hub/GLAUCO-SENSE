/* =========================================================
   GLAUCO-SENSE
   Main JavaScript
   Handles:
   - Navigation
   - Virtual Patient
   - IoT simulation
   - Live PPG
   - Signal processing
   - Feature calculations
   - What-If laboratory
   - Screening result
   - Complete demo
========================================================= */

// @ts-nocheck

document.addEventListener("DOMContentLoaded", () => {

    /* =====================================================
       GLOBAL STATE
    ===================================================== */

    const state = {

        currentSection: "overview",

        patientType: "reference",

        ppgRunning: false,

        ppgAnimationId: null,

        packetCount: 0,

        ppgTime: 0,

        processingRunning: false,

        demoRunning: false,

        demoTimer: null,

        measurementTimer: null,

        sbp: 118,

        dbp: 76,

        iop: 14.2,

        hr: 68

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

    const runDemoButton =
        document.getElementById("runDemoButton");


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


        /* Hide every page */

        pageSections.forEach(section => {

            section.classList.remove(
                "active-section"
            );

        });


        /* Show selected page */

        target.classList.add(
            "active-section"
        );


        /* Update navigation buttons */

        navButtons.forEach(button => {

            button.classList.remove("active");

            if (
                button.dataset.section ===
                sectionId
            ) {

                button.classList.add("active");

            }

        });


        state.currentSection =
            sectionId;


        /* Close mobile menu */

        if (window.innerWidth <= 700) {

            sidebar.classList.remove(
                "menu-open"
            );

        }


        /* Resize chart if monitor opened */

        if (sectionId === "monitor") {

            setTimeout(() => {

                resizePPGCanvas();

            }, 50);

        }


        /* Scroll page to top */

        window.scrollTo({

            top: 0,

            behavior: "smooth"

        });

    }


    /* =====================================================
       NAVIGATION CLICK EVENTS
    ===================================================== */

    navButtons.forEach(button => {

        button.addEventListener(
            "click",
            () => {

                const section =
                    button.dataset.section;

                navigateTo(section);

            }
        );

    });


    /* =====================================================
       MOBILE MENU
    ===================================================== */

    mobileMenuButton.addEventListener(
        "click",
        () => {

            sidebar.classList.toggle(
                "menu-open"
            );

        }
    );


    /* =====================================================
       PATIENT DATA
    ===================================================== */

    const patientData = {

        reference: {

            id: "SUBJ-SIM-1042",

            description:
                "Synthetic reference physiological profile.",

            age: 52,

            sex: "F",

            sqi: 98,

            sbp: 118,

            dbp: 76,

            hr: 68,

            iopRight: 14.2,

            iopLeft: 14.8,

            vcdr: 0.36

        },


        elevated: {

            id: "SUBJ-SIM-2087",

            description:
                "Synthetic profile demonstrating an elevated vascular-pattern simulation.",

            age: 61,

            sex: "M",

            sqi: 94,

            sbp: 148,

            dbp: 92,

            hr: 76,

            iopRight: 20.5,

            iopLeft: 21.1,

            vcdr: 0.54

        }

    };


    /* =====================================================
       HELPER FUNCTION
    ===================================================== */

    function setText(id, value) {

        const element =
            document.getElementById(id);

        if (element) {

            element.textContent =
                value;

        }

    }


    /* =====================================================
       PATIENT SELECTOR
    ===================================================== */

    const patientButtons =
        document.querySelectorAll(
            ".patient-select-button"
        );


    patientButtons.forEach(button => {

        button.addEventListener(
            "click",
            () => {

                const type =
                    button.dataset.patient;

                if (!patientData[type]) {

                    return;

                }

                state.patientType =
                    type;


                patientButtons.forEach(btn => {

                    btn.classList.remove(
                        "active"
                    );

                });


                button.classList.add(
                    "active"
                );


                loadPatient(type);

            }
        );

    });


    /* =====================================================
       LOAD PATIENT
    ===================================================== */

    function loadPatient(type) {

        const patient =
            patientData[type];


        state.sbp =
            patient.sbp;

        state.dbp =
            patient.dbp;

        state.hr =
            patient.hr;

        state.iop =
            patient.iopRight;


        /* Patient information */

        setText(
            "patientId",
            patient.id
        );

        setText(
            "patientDescription",
            patient.description
        );

        setText(
            "patientAge",
            patient.age
        );

        setText(
            "patientSex",
            patient.sex
        );

        setText(
            "patientSQI",
            patient.sqi + "%"
        );


        /* Patient measurements */

        setText(
            "patientSBP",
            patient.sbp
        );

        setText(
            "patientDBP",
            patient.dbp
        );

        setText(
            "patientHR",
            patient.hr
        );


        /* Table */

        setText(
            "tableSBP",
            patient.sbp + " mmHg"
        );

        setText(
            "tableDBP",
            patient.dbp + " mmHg"
        );

        setText(
            "tableHR",
            patient.hr + " bpm"
        );

        setText(
            "tableSQI",
            patient.sqi + "%"
        );


        /* Clinical reference */

        setText(
            "clinicalIOPR",
            patient.iopRight.toFixed(1)
        );

        setText(
            "clinicalIOPL",
            patient.iopLeft.toFixed(1)
        );

        setText(
            "clinicalVCDR",
            patient.vcdr.toFixed(2)
        );


        /* Update all calculated features */

        updateFeatureValues();

        updateLiveValues();

        updateResult();

        updateWhatIfFromPatient();

    }


    /* =====================================================
       SENSOR ACTIVATION
    ===================================================== */

    const activateSensorButton =
        document.getElementById(
            "activateSensorButton"
        );


    if (activateSensorButton) {

        activateSensorButton.addEventListener(
            "click",
            () => {

                activateSensorButton.textContent =
                    "✓ SENSOR SIMULATION ACTIVE";

                activateSensorButton.style.background =
                    "var(--green-800)";

                startPPG();

                navigateTo("monitor");

            }
        );

    }


    /* =====================================================
       MAP CALCULATION
    ===================================================== */

    function calculateMAP(sbp, dbp) {

        return (
            dbp +
            (sbp - dbp) / 3
        );

    }


    /* =====================================================
       MOPP CALCULATION
    ===================================================== */

    function calculateMOPP(
        sbp,
        dbp,
        iop
    ) {

        const map =
            calculateMAP(
                sbp,
                dbp
            );

        return (
            (2 / 3) * map -
            iop
        );

    }


    /* =====================================================
       DOPP CALCULATION
    ===================================================== */

    function calculateDOPP(
        dbp,
        iop
    ) {

        return dbp - iop;

    }


    /* =====================================================
       FEATURE VALUES
    ===================================================== */

    function updateFeatureValues() {

        const map =
            calculateMAP(
                state.sbp,
                state.dbp
            );


        const mopp =
            calculateMOPP(
                state.sbp,
                state.dbp,
                state.iop
            );


        const dopp =
            calculateDOPP(
                state.dbp,
                state.iop
            );


        setText(
            "featureSBP",
            Math.round(state.sbp)
        );

        setText(
            "featureDBP",
            Math.round(state.dbp)
        );

        setText(
            "featureMAP",
            map.toFixed(1)
        );

        setText(
            "featureHR",
            Math.round(state.hr)
        );

        setText(
            "featureMOPP",
            mopp.toFixed(1)
        );

        setText(
            "featureDOPP",
            dopp.toFixed(1)
        );

    }


    /* =====================================================
       LIVE VALUES
    ===================================================== */

    function updateLiveValues() {

        setText(
            "liveSBP",
            Math.round(state.sbp)
        );

        setText(
            "liveDBP",
            Math.round(state.dbp)
        );

        setText(
            "liveHR",
            Math.round(state.hr)
        );

    }


    /* =====================================================
       PPG CANVAS
    ===================================================== */

    const canvas =
        document.getElementById(
            "ppgCanvas"
        );

    const ctx =
        canvas ?
        canvas.getContext("2d") :
        null;


    function resizePPGCanvas() {

        if (!canvas || !ctx) {

            return;

        }


        const rect =
            canvas.getBoundingClientRect();


        if (
            rect.width === 0 ||
            rect.height === 0
        ) {

            return;

        }


        const dpr =
            window.devicePixelRatio ||
            1;


        canvas.width =
            rect.width * dpr;

        canvas.height =
            rect.height * dpr;


        ctx.setTransform(
            dpr,
            0,
            0,
            dpr,
            0,
            0
        );


        drawPPG();

    }


    window.addEventListener(
        "resize",
        resizePPGCanvas
    );


    /* =====================================================
       PPG WAVEFORM
    ===================================================== */

    function generatePPGValue(
        x,
        time
    ) {

        const frequency =
            state.hr / 60;


        const pulse =
            Math.sin(
                2 *
                Math.PI *
                frequency *
                time
            );


        const harmonic =
            0.25 *
            Math.sin(
                4 *
                Math.PI *
                frequency *
                time
            );


        const dicrotic =
            0.08 *
            Math.sin(
                7 *
                Math.PI *
                frequency *
                time
            );


        const noise =
            (
                Math.sin(
                    x * 0.13 +
                    time * 3
                ) *
                0.035
            );


        return (
            pulse +
            harmonic +
            dicrotic +
            noise
        );

    }


    /* =====================================================
       DRAW PPG
    ===================================================== */

    function drawPPG() {

        if (!canvas || !ctx) {

            return;

        }


        const width =
            canvas.clientWidth;

        const height =
            canvas.clientHeight;


        if (
            width <= 0 ||
            height <= 0
        ) {

            return;

        }


        /* Background */

        ctx.clearRect(
            0,
            0,
            width,
            height
        );


        ctx.fillStyle =
            "#fffdf8";

        ctx.fillRect(
            0,
            0,
            width,
            height
        );


        /* Grid */

        ctx.strokeStyle =
            "#e9dfca";

        ctx.lineWidth = 1;


        const gridX = 45;

        const gridY = 35;


        for (
            let x = 0;
            x < width;
            x += gridX
        ) {

            ctx.beginPath();

            ctx.moveTo(
                x,
                0
            );

            ctx.lineTo(
                x,
                height
            );

            ctx.stroke();

        }


        for (
            let y = 0;
            y < height;
            y += gridY
        ) {

            ctx.beginPath();

            ctx.moveTo(
                0,
                y
            );

            ctx.lineTo(
                width,
                y
            );

            ctx.stroke();

        }


        /* Wave */

        ctx.beginPath();

        ctx.strokeStyle =
            "#3d7b58";

        ctx.lineWidth = 2.2;


        const center =
            height / 2;


        const amplitude =
            height * 0.27;


        for (
            let x = 0;
            x <= width;
            x += 2
        ) {

            const time =
                x * 0.045 +
                state.ppgTime;


            const value =
                generatePPGValue(
                    x,
                    time
                );


            const y =
                center -
                value * amplitude;


            if (x === 0) {

                ctx.moveTo(
                    x,
                    y
                );

            } else {

                ctx.lineTo(
                    x,
                    y
                );

            }

        }


        ctx.stroke();


        /* Center line */

        ctx.beginPath();

        ctx.strokeStyle =
            "#cfc3b0";

        ctx.lineWidth = 1;

        ctx.setLineDash([
            5,
            5
        ]);

        ctx.moveTo(
            0,
            center
        );

        ctx.lineTo(
            width,
            center
        );

        ctx.stroke();

        ctx.setLineDash([]);


        /* Label */

        ctx.fillStyle =
            "#806957";

        ctx.font =
            "11px Arial";

        ctx.fillText(
            "SIMULATED PPG",
            12,
            20
        );

    }


    /* =====================================================
       PPG ANIMATION
    ===================================================== */

    function animatePPG() {

        if (!state.ppgRunning) {

            return;

        }


        state.ppgTime += 0.045;


        drawPPG();


        state.ppgAnimationId =
            requestAnimationFrame(
                animatePPG
            );

    }


    /* =====================================================
       START PPG
    ===================================================== */

    function startPPG() {

        if (state.ppgRunning) {

            return;

        }


        state.ppgRunning = true;


        if (state.currentSection === "monitor") {

            resizePPGCanvas();

        }


        animatePPG();


        startMeasurementCounter();

    }


    /* =====================================================
       PAUSE PPG
    ===================================================== */

    function pausePPG() {

        state.ppgRunning = false;


        if (
            state.ppgAnimationId
        ) {

            cancelAnimationFrame(
                state.ppgAnimationId
            );

            state.ppgAnimationId =
                null;

        }


        stopMeasurementCounter();

    }


    /* =====================================================
       RESET PPG
    ===================================================== */

    function resetPPG() {

        pausePPG();


        state.packetCount = 0;

        state.ppgTime = 0;


        setText(
            "packetCount",
            "0"
        );


        const progress =
            document.getElementById(
                "measurementProgress"
            );


        if (progress) {

            progress.style.width =
                "0%";

        }


        drawPPG();

    }


    /* =====================================================
       MEASUREMENT COUNTER
    ===================================================== */

    function startMeasurementCounter() {

        stopMeasurementCounter();


        state.measurementTimer =
            setInterval(
                () => {

                    state.packetCount += 1;


                    setText(
                        "packetCount",
                        state.packetCount
                    );


                    const progress =
                        Math.min(
                            state.packetCount %
                            101,
                            100
                        );


                    const progressElement =
                        document.getElementById(
                            "measurementProgress"
                        );


                    if (progressElement) {

                        progressElement.style.width =
                            progress + "%";

                    }


                    /* Slight natural variation */

                    state.hr =
                        patientData[
                            state.patientType
                        ].hr +
                        Math.sin(
                            state.packetCount *
                            0.15
                        ) *
                        2;


                    updateLiveValues();

                },
                500
            );

    }


    /* =====================================================
       STOP MEASUREMENT COUNTER
    ===================================================== */

    function stopMeasurementCounter() {

        if (
            state.measurementTimer
        ) {

            clearInterval(
                state.measurementTimer
            );

            state.measurementTimer =
                null;

        }

    }


    /* =====================================================
       MONITOR BUTTONS
    ===================================================== */

    const startPPGButton =
        document.getElementById(
            "startPPGButton"
        );


    const pausePPGButton =
        document.getElementById(
            "pausePPGButton"
        );


    const resetPPGButton =
        document.getElementById(
            "resetPPGButton"
        );


    if (startPPGButton) {

        startPPGButton.addEventListener(
            "click",
            startPPG
        );

    }


    if (pausePPGButton) {

        pausePPGButton.addEventListener(
            "click",
            pausePPG
        );

    }


    if (resetPPGButton) {

        resetPPGButton.addEventListener(
            "click",
            resetPPG
        );

    }


    /* =====================================================
       SIGNAL PROCESSING
    ===================================================== */

    const processSignalButton =
        document.getElementById(
            "processSignalButton"
        );


    const processingStages =
        document.querySelectorAll(
            ".processing-stage"
        );


    async function processSignal() {

        if (state.processingRunning) {

            return;

        }


        state.processingRunning = true;


        processSignalButton.disabled =
            true;

        processSignalButton.textContent =
            "⚙ Processing...";


        /* Reset stages */

        processingStages.forEach(
            stage => {

                stage.classList.remove(
                    "active"
                );

            }
        );


        /* Activate stages one by one */

        for (
            let i = 0;
            i < processingStages.length;
            i++
        ) {

            await wait(650);


            processingStages[
                i
            ].classList.add(
                "active"
            );

        }


        processSignalButton.textContent =
            "✓ Processing Complete";


        state.processingRunning =
            false;


        setTimeout(
            () => {

                processSignalButton.disabled =
                    false;

                processSignalButton.textContent =
                    "⚙ Process Signal";

            },
            1500
        );

    }


    if (processSignalButton) {

        processSignalButton.addEventListener(
            "click",
            processSignal
        );

    }


    /* =====================================================
       WAIT HELPER
    ===================================================== */

    function wait(milliseconds) {

        return new Promise(
            resolve =>
                setTimeout(
                    resolve,
                    milliseconds
                )
        );

    }


    /* =====================================================
       WHAT-IF CONTROLS
    ===================================================== */

    const whatIfSBP =
        document.getElementById(
            "whatIfSBP"
        );


    const whatIfDBP =
        document.getElementById(
            "whatIfDBP"
        );


    const whatIfIOP =
        document.getElementById(
            "whatIfIOP"
        );


    function updateWhatIf() {

        const sbp =
            Number(
                whatIfSBP.value
            );


        const dbp =
            Number(
                whatIfDBP.value
            );


        const iop =
            Number(
                whatIfIOP.value
            );


        const map =
            calculateMAP(
                sbp,
                dbp
            );


        const mopp =
            calculateMOPP(
                sbp,
                dbp,
                iop
            );


        const dopp =
            calculateDOPP(
                dbp,
                iop
            );


        setText(
            "whatIfSBPValue",
            sbp + " mmHg"
        );


        setText(
            "whatIfDBPValue",
            dbp + " mmHg"
        );


        setText(
            "whatIfIOPValue",
            iop.toFixed(1) +
            " mmHg"
        );


        setText(
            "whatIfMAP",
            map.toFixed(1)
        );


        setText(
            "whatIfMOPP",
            mopp.toFixed(1)
        );


        setText(
            "whatIfDOPP",
            dopp.toFixed(1)
        );


        const risk =
            calculateDemoRisk(
                sbp,
                dbp,
                iop
            );


        updateRiskGauge(
            risk
        );

    }


    if (whatIfSBP) {

        whatIfSBP.addEventListener(
            "input",
            updateWhatIf
        );

    }


    if (whatIfDBP) {

        whatIfDBP.addEventListener(
            "input",
            updateWhatIf
        );

    }


    if (whatIfIOP) {

        whatIfIOP.addEventListener(
            "input",
            updateWhatIf
        );

    }


    /* =====================================================
       WHAT-IF FROM PATIENT
    ===================================================== */

    function updateWhatIfFromPatient() {

        if (!whatIfSBP) {

            return;

        }


        whatIfSBP.value =
            state.sbp;

        whatIfDBP.value =
            state.dbp;

        whatIfIOP.value =
            state.iop;


        updateWhatIf();

    }


    /* =====================================================
       DEMONSTRATION HEURISTIC
       NOT A CLINICAL MODEL
    ===================================================== */

    function calculateDemoRisk(
        sbp,
        dbp,
        iop
    ) {

        let score = 0;


        /* BP contribution */

        if (sbp >= 140) {

            score += 0.30;

        } else if (sbp >= 130) {

            score += 0.15;

        }


        if (dbp >= 90) {

            score += 0.25;

        } else if (dbp >= 80) {

            score += 0.12;

        }


        /* IOP contribution */

        if (iop >= 21) {

            score += 0.30;

        } else if (iop >= 18) {

            score += 0.15;

        }


        return Math.min(
            score,
            1
        );

    }


    /* =====================================================
       RISK GAUGE
    ===================================================== */

    function updateRiskGauge(
        risk
    ) {

        const gauge =
            document.getElementById(
                "riskGauge"
            );


        if (gauge) {

            gauge.style.width =
                (risk * 100) + "%";


            if (risk < 0.35) {

                gauge.style.background =
                    "var(--green-600)";

            } else if (risk < 0.65) {

                gauge.style.background =
                    "#a38a48";

            } else {

                gauge.style.background =
                    "#9a5a4d";

            }

        }


        setText(
            "whatIfRisk",
            risk.toFixed(2)
        );

    }


    /* =====================================================
       SCREENING RESULT
    ===================================================== */

    function updateResult() {

        const risk =
            calculateDemoRisk(
                state.sbp,
                state.dbp,
                state.iop
            );


        const title =
            document.getElementById(
                "resultTitle"
            );


        const description =
            document.getElementById(
                "resultDescription"
            );


        if (risk >= 0.5) {

            setText(
                "resultTitle",
                "ELEVATED SCREENING CONCERN"
            );


            setText(
                "resultDescription",
                "The synthetic profile produces an elevated pattern in this demonstration. This is not a diagnosis."
            );


            if (title) {

                title.classList.remove(
                    "green-text"
                );

                title.classList.add(
                    "yellow-text"
                );

            }

        } else {

            setText(
                "resultTitle",
                "LOWER SCREENING CONCERN"
            );


            setText(
                "resultDescription",
                "The current synthetic reference profile does not produce an elevated concern in this demonstration."
            );


            if (title) {

                title.classList.remove(
                    "yellow-text"
                );

                title.classList.add(
                    "green-text"
                );

            }

        }


        setText(
            "resultBP",
            risk >= 0.5
                ? "Elevated pattern"
                : "Reference"
        );


        setText(
            "resultIOP",
            state.iop.toFixed(1)
        );


        setText(
            "resultScore",
            risk.toFixed(2)
        );

    }


    /* =====================================================
       MODEL SELECTOR
    ===================================================== */

    const modelSelect =
        document.getElementById(
            "modelSelect"
        );


    if (modelSelect) {

        modelSelect.addEventListener(
            "change",
            () => {

                console.log(
                    "Selected research model:",
                    modelSelect.value
                );

            }
        );

    }


    /* =====================================================
       COMPLETE DEMO
    ===================================================== */

    const demoSequence = [

        "overview",

        "patient",

        "iot",

        "monitor",

        "signal",

        "clinical",

        "features",

        "model",

        "result",

        "whatif",

        "architecture",

        "hardware",

        "research"

    ];


    async function runCompleteDemo() {

        if (state.demoRunning) {

            return;

        }


        state.demoRunning = true;


        runDemoButton.disabled =
            true;

        runDemoButton.textContent =
            "⏳ DEMO RUNNING";


        /* Start with reference patient */

        const referenceButton =
            document.querySelector(
                '[data-patient="reference"]'
            );


        if (referenceButton) {

            referenceButton.click();

        }


        for (
            let i = 0;
            i < demoSequence.length;
            i++
        ) {

            const section =
                demoSequence[i];


            navigateTo(section);


            /* Run special actions */

            if (
                section === "monitor"
            ) {

                startPPG();

                await wait(2500);

                pausePPG();

            }


            if (
                section === "signal"
            ) {

                await processSignal();

            }


            if (
                section === "whatif"
            ) {

                await wait(1200);

            }


            await wait(900);

        }


        navigateTo("overview");


        runDemoButton.disabled =
            false;

        runDemoButton.textContent =
            "▶ RUN COMPLETE DEMO";


        state.demoRunning =
            false;

    }


    if (runDemoButton) {

        runDemoButton.addEventListener(
            "click",
            runCompleteDemo
        );

    }


    /* =====================================================
       INITIALIZATION
    ===================================================== */

    function initializeApp() {

        /* Load default patient */

        loadPatient(
            "reference"
        );


        /* Draw initial PPG */

        setTimeout(
            () => {

                resizePPGCanvas();

            },
            100
        );


        /* Initialize What-If */

        updateWhatIf();


        /* Initialize result */

        updateResult();


        /* Start on overview */

        navigateTo(
            "overview"
        );

    }


    initializeApp();


    /* =====================================================
       CLEANUP
    ===================================================== */

    window.addEventListener(
        "beforeunload",
        () => {

            pausePPG();

            stopMeasurementCounter();

            if (state.demoTimer) {

                clearTimeout(
                    state.demoTimer
                );

            }

        }
    );

});