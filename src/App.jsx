import Webcam from "react-webcam";
import { useEffect, useMemo, useRef, useState } from "react";
import { pipeline } from "@huggingface/transformers";
import emailjs from "@emailjs/browser";

const HIGH_CONFIDENCE = 80;
const LOW_CONFIDENCE = 75;
const SPREAD_THRESHOLD = 6;

const NEGATIVE_FEELINGS = ["Worried", "Sad", "Angry"];
const MANUAL_CHOICES_COUNT = 6;
const MAX_RECOGNITION_ATTEMPTS = 3;
const SUCCESS_RESET_MS = 3000;

const FEELINGS = [
  { label: "Great", emoji: "😁" },
  { label: "Happy", emoji: "😊" },
  { label: "Okay", emoji: "🙂" },
  { label: "Tired", emoji: "😴" },
  { label: "Excited", emoji: "🤩" },
  { label: "Worried", emoji: "😟" },
  { label: "Sad", emoji: "😢" },
  { label: "Angry", emoji: "😠" },
];

const EMAILJS_SERVICE_ID = "service_hljpvvd";
const EMAILJS_TEMPLATE_ID = "template_sqifik5";
const EMAILJS_PUBLIC_KEY = "VxCZ-xTxfOwF6fzzw";

const TEACHER_PASSCODE = "1234";

const studentModules = import.meta.glob("./students/*.{jpg,jpeg,png}", {
  eager: true,
  import: "default",
});

const STUDENTS = Object.entries(studentModules)
  .map(([path, image]) => {
    const file = path.split("/").pop();
    const fileName = file.replace(/\.[^.]+$/, "");

    return {
      id: fileName.toLowerCase(),
      name: fileName.replace(/[_-]/g, " "),
      image,
    };
  })
  .sort((a, b) => a.name.localeCompare(b.name));

function cosineSimilarity(a, b) {
  if (!a || !b || a.length !== b.length) return 0;

  let dot = 0;
  let magA = 0;
  let magB = 0;

  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    magA += a[i] * a[i];
    magB += b[i] * b[i];
  }

  const denom = Math.sqrt(magA) * Math.sqrt(magB);
  return denom ? dot / denom : 0;
}

function toArray(output) {
  if (!output) return [];
  if (output.data) return Array.from(output.data);
  if (Array.isArray(output)) return output.flat(Infinity).map(Number);
  if (output.tolist) return output.tolist().flat(Infinity).map(Number);
  return [];
}

function getTodayKey() {
  return new Date().toISOString().split("T")[0];
}

function isSameDayCheckIn(checkIn) {
  return checkIn?.dateKey === getTodayKey();
}

function getConfidenceLevel(bestMatch, possibleMismatch, manualCorrectionUsed) {
  if (manualCorrectionUsed) return "low";
  if (!bestMatch) return "low";

  if (bestMatch.percentage >= 85 && !possibleMismatch) return "high";
  if (bestMatch.percentage >= 75) return "medium";
  return "low";
}

const confidenceStyles = {
  high: {
    bg: "#e8f5e9",
    text: "#2e7d32",
    label: "✓ We think this is you",
  },
  medium: {
    bg: "#fff8e1",
    text: "#9b6b00",
    label: "Please double check",
  },
  low: {
    bg: "#ffebee",
    text: "#c62828",
    label: "Not sure — please check",
  },
};

function getDashboardStats(checkIns) {
  const today = getTodayKey();
  const todayCheckIns = checkIns.filter((c) => c.dateKey === today);

  const feelingsCount = {};
  todayCheckIns.forEach((c) => {
    feelingsCount[c.feeling] = (feelingsCount[c.feeling] || 0) + 1;
  });

  return {
    total: todayCheckIns.length,
    followUp: todayCheckIns.filter((c) => c.needsFollowUp).length,
    manual: todayCheckIns.filter((c) => c.manualCorrectionUsed).length,
    mismatch: todayCheckIns.filter((c) => c.possibleMismatch).length,
    feelingsCount,
  };
}

const squareImageStyle = {
  width: "100%",
  aspectRatio: "1 / 1",
  objectFit: "cover",
  borderRadius: 12,
};

function addHoverHandlers(baseColor = "white", hoverColor = "#f5f5f5") {
  return {
    onMouseOver: (e) => {
      e.currentTarget.style.background = hoverColor;
    },
    onMouseOut: (e) => {
      e.currentTarget.style.background = baseColor;
    },
  };
}

export default function App() {
  const webcamRef = useRef(null);
  const resetTimerRef = useRef(null);

  const [windowWidth, setWindowWidth] = useState(
    typeof window !== "undefined" ? window.innerWidth : 1200
  );

  const [extractor, setExtractor] = useState(null);
  const [studentEmbeddings, setStudentEmbeddings] = useState({});
  const [loadingModel, setLoadingModel] = useState(true);
  const [status, setStatus] = useState("Loading AI model...");
  const [error, setError] = useState("");

  const [capturedImage, setCapturedImage] = useState(null);
  const [results, setResults] = useState([]);
  const [matchedUser, setMatchedUser] = useState(null);
  const [selectedFeeling, setSelectedFeeling] = useState("");
  const [checkInSubmitted, setCheckInSubmitted] = useState(false);
  const [successMessage, setSuccessMessage] = useState(false);

  const [allCheckIns, setAllCheckIns] = useState(() => {
    const saved = localStorage.getItem("classCheckIns");
    return saved ? JSON.parse(saved) : [];
  });

  const [teacherPanelOpen, setTeacherPanelOpen] = useState(false);
  const [teacherUnlocked, setTeacherUnlocked] = useState(false);
  const [teacherPasscodeInput, setTeacherPasscodeInput] = useState("");
  const [teacherPasscodeError, setTeacherPasscodeError] = useState("");

  const [duplicateMessage, setDuplicateMessage] = useState("");
  const [alreadyCheckedInToday, setAlreadyCheckedInToday] = useState(false);
  const [possibleMismatch, setPossibleMismatch] = useState(false);

  const [manualCorrectionUsed, setManualCorrectionUsed] = useState(false);
  const [autoMatchedUser, setAutoMatchedUser] = useState(null);

  const [recognitionAttempts, setRecognitionAttempts] = useState(0);

  // selector modes: "none" | "top6" | "full"
  const [selectionMode, setSelectionMode] = useState("none");
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedLetter, setSelectedLetter] = useState("");

  const alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");

  const bestMatch = useMemo(() => results[0] || null, [results]);
  const secondBestMatch = useMemo(() => results[1] || null, [results]);
  const thirdBestMatch = useMemo(() => results[2] || null, [results]);
  const topManualChoices = useMemo(
    () => results.slice(0, MANUAL_CHOICES_COUNT),
    [results]
  );

  const displayMatch =
    matchedUser || (selectionMode === "full" ? null : bestMatch);

  const confidenceLevel = getConfidenceLevel(
    bestMatch,
    possibleMismatch,
    manualCorrectionUsed
  );
  const confidence = confidenceStyles[confidenceLevel];
  const dashboardStats = getDashboardStats(allCheckIns);

  const studentsByLetter = useMemo(() => {
    const grouped = {};
    STUDENTS.forEach((student) => {
      const firstLetter = student.name.charAt(0).toUpperCase();
      if (!grouped[firstLetter]) grouped[firstLetter] = [];
      grouped[firstLetter].push(student);
    });
    return grouped;
  }, []);

  const filteredStudents = useMemo(() => {
    const query = searchQuery.trim().toLowerCase();

    if (query) {
      return STUDENTS.filter((student) =>
        student.name.toLowerCase().includes(query)
      );
    }

    if (selectedLetter) {
      return studentsByLetter[selectedLetter] || [];
    }

    return [];
  }, [searchQuery, selectedLetter, studentsByLetter]);

  const isSingleColumn = windowWidth < 980;
  const isNarrow = windowWidth < 720;
  const feelingsColumns = windowWidth < 720 ? 2 : 4;
  const selectorColumns = windowWidth < 720 ? 2 : 3;
  const letterColumns = windowWidth < 720 ? 5 : 7;

  useEffect(() => {
    function handleResize() {
      setWindowWidth(window.innerWidth);
    }

    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  useEffect(() => {
    let cancelled = false;

    async function loadEverything() {
      try {
        setError("");
        setStatus("Loading AI model...");

        const imageExtractor = await pipeline(
          "image-feature-extraction",
          "Xenova/clip-vit-base-patch32"
        );

        if (cancelled) return;
        setExtractor(() => imageExtractor);

        setStatus("Preparing student photo library...");

        const cache = {};
        for (const student of STUDENTS) {
          const output = await imageExtractor(student.image, {
            pooling: "mean",
            normalize: true,
          });
          cache[student.id] = toArray(output);
        }

        if (cancelled) return;
        setStudentEmbeddings(cache);
        setStatus("Ready");
      } catch (err) {
        console.error(err);
        setError("Could not load the AI model or student images.");
        setStatus("");
      } finally {
        if (!cancelled) setLoadingModel(false);
      }
    }

    loadEverything();

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    localStorage.setItem("classCheckIns", JSON.stringify(allCheckIns));
  }, [allCheckIns]);

  useEffect(() => {
    return () => {
      if (resetTimerRef.current) clearTimeout(resetTimerRef.current);
    };
  }, []);

  async function sendTeacherAlertEmail(checkIn) {
    try {
      await emailjs.send(
        EMAILJS_SERVICE_ID,
        EMAILJS_TEMPLATE_ID,
        {
          student_name: checkIn.name,
          feeling: checkIn.feeling,
          time: checkIn.time,
        },
        {
          publicKey: EMAILJS_PUBLIC_KEY,
        }
      );
      console.log("Teacher alert email sent");
    } catch (error) {
      console.error("Email alert failed:", error);
    }
  }

  function playSuccessSound() {
    try {
      const audioContext = new (window.AudioContext ||
        window.webkitAudioContext)();
      const oscillator = audioContext.createOscillator();
      const gainNode = audioContext.createGain();

      oscillator.type = "sine";
      oscillator.frequency.setValueAtTime(880, audioContext.currentTime);
      oscillator.frequency.setValueAtTime(
        1174,
        audioContext.currentTime + 0.12
      );

      gainNode.gain.setValueAtTime(0.0001, audioContext.currentTime);
      gainNode.gain.exponentialRampToValueAtTime(
        0.15,
        audioContext.currentTime + 0.02
      );
      gainNode.gain.exponentialRampToValueAtTime(
        0.0001,
        audioContext.currentTime + 0.3
      );

      oscillator.connect(gainNode);
      gainNode.connect(audioContext.destination);

      oscillator.start();
      oscillator.stop(audioContext.currentTime + 0.3);
    } catch (error) {
      console.error("Sound failed:", error);
    }
  }

  function resetSelectionUi() {
    setSelectionMode("none");
    setSearchQuery("");
    setSelectedLetter("");
  }

  async function captureAndMatch() {
    try {
      if (!webcamRef.current || !extractor) return;

      setSuccessMessage(false);
      if (resetTimerRef.current) clearTimeout(resetTimerRef.current);

      const screenshot = webcamRef.current.getScreenshot();
      if (!screenshot) return;

      setCapturedImage(screenshot);
      setResults([]);
      setSelectedFeeling("");
      setCheckInSubmitted(false);
      setDuplicateMessage("");
      setAlreadyCheckedInToday(false);
      setPossibleMismatch(false);
      setManualCorrectionUsed(false);
      setAutoMatchedUser(null);
      setMatchedUser(null);
      resetSelectionUi();
      setStatus("Matching...");

      const webcamOutput = await extractor(screenshot, {
        pooling: "mean",
        normalize: true,
      });

      const webcamEmbedding = toArray(webcamOutput);

      const ranked = STUDENTS.map((student) => {
        const score = cosineSimilarity(
          webcamEmbedding,
          studentEmbeddings[student.id]
        );

        return {
          ...student,
          score,
          percentage: Math.round(((score + 1) / 2) * 100),
        };
      }).sort((a, b) => b.score - a.score);

      setResults(ranked);

      const best = ranked[0];
      const second = ranked[1];
      const third = ranked[2];
      const spread = best && third ? best.percentage - third.percentage : 100;

      if (best && second && best.percentage - second.percentage <= 3) {
        setPossibleMismatch(true);
      }

      // CASE 1: High confidence -> auto-match
      if (
        best &&
        best.percentage >= HIGH_CONFIDENCE &&
        spread >= SPREAD_THRESHOLD
      ) {
        setMatchedUser(best);
        setAutoMatchedUser(best);
        setRecognitionAttempts(0);
        setSelectionMode("none");

        const alreadyChecked = allCheckIns.some(
          (item) => item.studentId === best.id && isSameDayCheckIn(item)
        );

        if (alreadyChecked) {
          setAlreadyCheckedInToday(true);
          setDuplicateMessage(`${best.name} has already checked in today.`);
        }

        setStatus("Ready");
        return;
      }

      // CASE 2: Medium confidence -> top 6 first
      if (best && best.percentage >= LOW_CONFIDENCE) {
        setAutoMatchedUser(best);
        setSelectionMode("top6");
        setRecognitionAttempts(0);
        setStatus("Choose your photo");
        return;
      }

      // CASE 3: Low confidence -> full selector
      const nextAttempts = recognitionAttempts + 1;
      setRecognitionAttempts(nextAttempts);

      if (nextAttempts >= MAX_RECOGNITION_ATTEMPTS) {
        setSelectionMode("full");
        setAutoMatchedUser(best || null);
        setStatus("Search or choose your name");
      } else {
        setStatus("Ready");
      }
    } catch (err) {
      console.error(err);
      setError("Match failed.");
      setStatus("");
    }
  }

  function chooseManualMatch(student) {
    setMatchedUser(student);
    setManualCorrectionUsed(true);
    resetSelectionUi();
    setRecognitionAttempts(0);

    const alreadyChecked = allCheckIns.some(
      (item) => item.studentId === student.id && isSameDayCheckIn(item)
    );

    if (alreadyChecked) {
      setAlreadyCheckedInToday(true);
      setDuplicateMessage(`${student.name} has already checked in today.`);
    } else {
      setAlreadyCheckedInToday(false);
      setDuplicateMessage("");
    }
  }

  function clearCurrentSession() {
    setCapturedImage(null);
    setResults([]);
    setMatchedUser(null);
    setSelectedFeeling("");
    setCheckInSubmitted(false);
    setDuplicateMessage("");
    setAlreadyCheckedInToday(false);
    setPossibleMismatch(false);
    setManualCorrectionUsed(false);
    setAutoMatchedUser(null);
    setRecognitionAttempts(0);
    resetSelectionUi();
  }

  function submitFeeling(feeling) {
    setSelectedFeeling(feeling);
    setCheckInSubmitted(true);
    setDuplicateMessage("");

    if (!matchedUser) return;

    const alreadyCheckedIn = allCheckIns.some(
      (item) => item.studentId === matchedUser.id && isSameDayCheckIn(item)
    );

    if (alreadyCheckedIn) {
      setAlreadyCheckedInToday(true);
      setDuplicateMessage(`${matchedUser.name} has already checked in today.`);
      return;
    }

    const checkIn = {
      id: crypto.randomUUID(),
      name: matchedUser.name,
      studentId: matchedUser.id,
      feeling,
      time: new Date().toLocaleString(),
      dateKey: getTodayKey(),
      needsFollowUp: NEGATIVE_FEELINGS.includes(feeling),
      capturedImage,
      bestMatchName: bestMatch ? bestMatch.name : "",
      bestMatchScore: bestMatch ? bestMatch.percentage : null,
      secondBestMatchName: secondBestMatch ? secondBestMatch.name : "",
      secondBestMatchScore: secondBestMatch ? secondBestMatch.percentage : null,
      possibleMismatch,
      autoMatchedName: autoMatchedUser ? autoMatchedUser.name : "",
      finalChosenName: matchedUser ? matchedUser.name : "",
      manualCorrectionUsed,
    };

    setAllCheckIns((prev) => [checkIn, ...prev]);

    if (checkIn.needsFollowUp) {
      sendTeacherAlertEmail(checkIn);
    }

    playSuccessSound();
    setSuccessMessage(true);
    clearCurrentSession();
    setStatus("Ready for next person");

    resetTimerRef.current = setTimeout(() => {
      setSuccessMessage(false);
      setStatus("Ready");
    }, SUCCESS_RESET_MS);
  }

  function deleteSingleCheckIn(id) {
    setAllCheckIns((prev) => prev.filter((item) => item.id !== id));
  }

  function unlockTeacherView() {
    if (teacherPasscodeInput === TEACHER_PASSCODE) {
      setTeacherUnlocked(true);
      setTeacherPasscodeError("");
      setTeacherPasscodeInput("");
    } else {
      setTeacherPasscodeError("Incorrect passcode");
    }
  }

  function lockTeacherView() {
    setTeacherUnlocked(false);
    setTeacherPasscodeInput("");
    setTeacherPasscodeError("");
  }

  return (
    <div
      style={{
        fontFamily: '"Edu SA Beginner", cursive',
        fontOpticalSizing: "auto",
        fontWeight: 500,
        fontStyle: "normal",
        padding: 16,
        background: "#f4f4f4",
        minHeight: "100vh",
      }}
    >
      <div style={{ maxWidth: 1080, margin: "0 auto", position: "relative" }}>
        <div className="logo">
  <img src="/logo.png" alt="School logo" />
</div>

        <div style={{ paddingTop: 8 }}>
          <h1
            style={{
              textAlign: "center",
              marginBottom: 6,
              fontSize: 34,
              lineHeight: 1.1,
            }}
          >
            Y4 Morning Check In
          </h1>

          <p
            style={{
              textAlign: "center",
              color: "#666",
              fontSize: 16,
              marginTop: 0,
              marginBottom: 10,
            }}
          >
            Look at the camera to check in for the day.
          </p>
        </div>

        {status && (
          <div
            style={{
              background: "#eef3ff",
              padding: "8px 12px",
              borderRadius: 10,
              marginBottom: 10,
              textAlign: "center",
              transition: "opacity 0.25s ease",
              fontSize: 16,
            }}
          >
            {status}
          </div>
        )}

        {error && (
          <div
            style={{
              background: "#fff0f0",
              color: "#a00000",
              padding: 10,
              borderRadius: 10,
              marginBottom: 10,
            }}
          >
            {error}
          </div>
        )}

        <div
          style={{
            display: "grid",
            gridTemplateColumns: isSingleColumn ? "1fr" : "1fr 1fr",
            gap: 16,
            alignItems: "start",
          }}
        >
          <div
            style={{
              background: "white",
              padding: 12,
              borderRadius: 16,
              boxShadow: "0 4px 12px rgba(0,0,0,0.04)",
            }}
          >
            <h2 style={{ textAlign: "center", fontSize: 22, margin: "4px 0 10px" }}>
              Camera
            </h2>

            <Webcam
              ref={webcamRef}
              audio={false}
              screenshotFormat="image/jpeg"
              videoConstraints={{
                facingMode: "user",
                aspectRatio: 1,
              }}
              style={{
                width: "100%",
                aspectRatio: "1 / 1",
                objectFit: "cover",
                borderRadius: 12,
              }}
            />

            <button
              onClick={captureAndMatch}
              disabled={
                loadingModel ||
                !extractor ||
                Object.keys(studentEmbeddings).length === 0
              }
              style={{
                marginTop: 12,
                padding: "10px 14px",
                borderRadius: 10,
                border: "none",
                background: "black",
                color: "white",
                cursor: "pointer",
                display: "block",
                marginLeft: "auto",
                marginRight: "auto",
                fontSize: 15,
                fontWeight: 600,
                fontFamily: '"Edu SA Beginner", cursive',
              }}
            >
              Tap to Start
            </button>

            {recognitionAttempts > 0 && selectionMode === "none" && !matchedUser && (
              <div
                style={{
                  marginTop: 8,
                  fontSize: 13,
                  color: "#555",
                  textAlign: "center",
                }}
              >
                Attempt {recognitionAttempts} of {MAX_RECOGNITION_ATTEMPTS}
              </div>
            )}
          </div>

          <div
            style={{
              background: "white",
              padding: 12,
              borderRadius: 16,
              boxShadow: "0 4px 12px rgba(0,0,0,0.04)",
            }}
          >
            <h2 style={{ textAlign: "center", fontSize: 22, margin: "4px 0 10px" }}>
              Result
            </h2>

            {!capturedImage && !successMessage && (
              <p style={{ textAlign: "center", color: "#666", margin: "10px 0" }}>
                No image captured yet.
              </p>
            )}

            {!capturedImage && successMessage && (
              <div
                style={{
                  marginTop: 8,
                  padding: 16,
                  background: "#eaf8ea",
                  borderRadius: 12,
                  textAlign: "center",
                  animation: "fadeIn 0.35s ease",
                }}
              >
                <h2 style={{ marginTop: 0, marginBottom: 8, fontSize: 24 }}>
                  ✅ Check-in complete
                </h2>
                <p style={{ margin: 0, color: "#555", fontSize: 15 }}>
                  Next person: step up to the camera and press{" "}
                  <strong>Tap to start</strong>.
                </p>
              </div>
            )}

            {capturedImage && (
              <>
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "1fr 1fr",
                    gap: 10,
                  }}
                >
                  <div>
                    <h3 style={{ textAlign: "center", margin: "0 0 8px", fontSize: 16 }}>
                      You
                    </h3>
                    <img
                      src={capturedImage}
                      alt="Captured"
                      style={squareImageStyle}
                    />
                  </div>

                  <div>
                    <h3 style={{ textAlign: "center", margin: "0 0 8px", fontSize: 16 }}>
                      Match
                    </h3>
                    {displayMatch ? (
                      <img
                        src={displayMatch.image}
                        alt={displayMatch.name}
                        style={squareImageStyle}
                      />
                    ) : (
                      <div
                        style={{
                          ...squareImageStyle,
                          background: "#f2f2f2",
                          display: "grid",
                          placeItems: "center",
                          color: "#666",
                          fontSize: 14,
                        }}
                      >
                        {selectionMode === "full" ? "Choose your name" : "Waiting for match"}
                      </div>
                    )}
                  </div>
                </div>

                {displayMatch && (
                  <div
                    style={{
                      marginTop: 12,
                      padding: 14,
                      background: confidence.bg,
                      borderRadius: 12,
                      textAlign: "center",
                      transition: "all 0.2s ease",
                    }}
                  >
                    <div
                      style={{
                        fontSize: 24,
                        fontWeight: 700,
                        lineHeight: 1.1,
                      }}
                    >
                      {displayMatch.name}
                    </div>

                    <div
  style={{
    marginTop: 8,
    fontSize: 16,
    fontWeight: 600,
    color: confidence.text,
  }}
>
  {confidenceLevel === "high" && "✓ You're all set"}
  {confidenceLevel === "medium" && "Looks like you"}
  {confidenceLevel === "low" && "Please check carefully"}
</div>

                    {possibleMismatch && !manualCorrectionUsed && (
  <div
    style={{
      marginTop: 8,
      color: "#c62828",
      fontWeight: "bold",
      fontSize: 14,
    }}
  >
    ⚠️ Please double check this is you
  </div>
)}

                    {manualCorrectionUsed && (
                      <div
                        style={{
                          marginTop: 8,
                          fontSize: 14,
                          fontWeight: "bold",
                          color: "#555",
                        }}
                      >
                        Selected manually
                      </div>
                    )}
                  </div>
                )}

                {selectionMode === "top6" && !matchedUser && (
                  <div
                    style={{
                      marginTop: 12,
                      padding: 10,
                      background: "#fff7e6",
                      borderRadius: 12,
                      color: "#8a5a00",
                      fontWeight: "bold",
                      textAlign: "center",
                      fontSize: 14,
                    }}
                  >
                    But... we are not fully sure. Choose your photo below.
                  </div>
                )}

                {selectionMode === "top6" && !matchedUser && (
                  <div
                    style={{
                      marginTop: 12,
                      padding: 12,
                      background: "#f7f7f7",
                      borderRadius: 12,
                    }}
                  >
                    <p
                      style={{
                        fontWeight: "bold",
                        marginTop: 0,
                        marginBottom: 10,
                        textAlign: "center",
                      }}
                    >
                      Top matches:
                    </p>
                    <div
                      style={{
                        display: "grid",
                        gridTemplateColumns: `repeat(${selectorColumns}, 1fr)`,
                        gap: 10,
                      }}
                    >
                      {topManualChoices.map((student) => (
                        <button
                          key={student.id}
                          onClick={() => chooseManualMatch(student)}
                          style={{
                            border: "1px solid #ddd",
                            borderRadius: 12,
                            background: "white",
                            padding: 8,
                            cursor: "pointer",
                            fontFamily: '"Edu SA Beginner", cursive',
                          }}
                          {...addHoverHandlers("white", "#f5f5f5")}
                        >
                          <img
                            src={student.image}
                            alt={student.name}
                            style={{
                              width: "100%",
                              aspectRatio: "1 / 1",
                              objectFit: "cover",
                              borderRadius: 8,
                              marginBottom: 8,
                            }}
                          />
                          <div style={{ fontWeight: "bold", fontSize: 14 }}>
                            {student.name}
                          </div>
                          <div style={{ fontSize: 12 }}>{student.percentage}%</div>
                        </button>
                      ))}
                    </div>

                    <div
                      style={{
                        display: "flex",
                        gap: 10,
                        justifyContent: "center",
                        flexWrap: "wrap",
                        marginTop: 12,
                      }}
                    >
                      <button
                        onClick={() => {
                          setSelectionMode("full");
                          setSearchQuery("");
                          setSelectedLetter("");
                        }}
                        style={{
                          padding: "8px 12px",
                          borderRadius: 10,
                          border: "1px solid #ccc",
                          cursor: "pointer",
                          background: "white",
                          fontFamily: '"Edu SA Beginner", cursive',
                        }}
                        {...addHoverHandlers("white", "#f5f5f5")}
                      >
                        None of these are me
                      </button>

                      <button
                        onClick={() => {
                          resetSelectionUi();
                          setStatus("Ready");
                        }}
                        style={{
                          padding: "8px 12px",
                          borderRadius: 10,
                          border: "1px solid #ccc",
                          cursor: "pointer",
                          background: "white",
                          fontFamily: '"Edu SA Beginner", cursive',
                        }}
                        {...addHoverHandlers("white", "#f5f5f5")}
                      >
                        Cancel
                      </button>
                    </div>
                  </div>
                )}

                {selectionMode === "full" && !matchedUser && (
                  <div
                    style={{
                      marginTop: 12,
                      padding: 12,
                      background: "#f7f7f7",
                      borderRadius: 12,
                    }}
                  >
                    <p
                      style={{
                        fontWeight: "bold",
                        marginTop: 0,
                        marginBottom: 10,
                        textAlign: "center",
                      }}
                    >
                      Choose your name:
                    </p>

                    <input
                      type="text"
                      value={searchQuery}
                      onChange={(e) => {
                        setSearchQuery(e.target.value);
                        if (e.target.value) setSelectedLetter("");
                      }}
                      placeholder="Type your name"
                      style={{
                        width: "100%",
                        padding: "10px 12px",
                        borderRadius: 10,
                        border: "1px solid #ccc",
                        marginBottom: 10,
                        fontFamily: '"Edu SA Beginner", cursive',
                        fontSize: 15,
                        boxSizing: "border-box",
                      }}
                    />

                    <div
                      style={{
                        display: "grid",
                        gridTemplateColumns: `repeat(${letterColumns}, 1fr)`,
                        gap: 6,
                        marginBottom: 10,
                      }}
                    >
                      {alphabet.map((letter) => (
                        <button
                          key={letter}
                          onClick={() => {
                            setSelectedLetter(letter);
                            setSearchQuery("");
                          }}
                          style={{
                            padding: "7px",
                            borderRadius: 8,
                            border: "1px solid #ccc",
                            background:
                              selectedLetter === letter ? "black" : "white",
                            color:
                              selectedLetter === letter ? "white" : "black",
                            cursor: "pointer",
                            fontFamily: '"Edu SA Beginner", cursive',
                          }}
                        >
                          {letter}
                        </button>
                      ))}
                    </div>

                    {filteredStudents.length > 0 ? (
                      <div
                        style={{
                          display: "grid",
                          gridTemplateColumns: `repeat(${selectorColumns}, 1fr)`,
                          gap: 10,
                        }}
                      >
                        {filteredStudents.map((student) => (
                          <button
                            key={student.id}
                            onClick={() => chooseManualMatch(student)}
                            style={{
                              border: "1px solid #ddd",
                              borderRadius: 12,
                              background: "white",
                              padding: 8,
                              cursor: "pointer",
                              fontFamily: '"Edu SA Beginner", cursive',
                            }}
                            {...addHoverHandlers("white", "#f5f5f5")}
                          >
                            <img
                              src={student.image}
                              alt={student.name}
                              style={{
                                width: "100%",
                                aspectRatio: "1 / 1",
                                objectFit: "cover",
                                borderRadius: 8,
                                marginBottom: 8,
                              }}
                            />
                            <div style={{ fontWeight: "bold", fontSize: 14 }}>
                              {student.name}
                            </div>
                          </button>
                        ))}
                      </div>
                    ) : (
                      <div
                        style={{
                          textAlign: "center",
                          color: "#666",
                          padding: "8px 0",
                          fontSize: 14,
                        }}
                      >
                        {searchQuery || selectedLetter
                          ? "No names found."
                          : "Choose a letter or type a name."}
                      </div>
                    )}

                    <div
                      style={{
                        display: "flex",
                        justifyContent: "center",
                        marginTop: 12,
                      }}
                    >
                      <button
                        onClick={() => {
                          resetSelectionUi();
                          setStatus("Ready");
                        }}
                        style={{
                          padding: "8px 12px",
                          borderRadius: 10,
                          border: "1px solid #ccc",
                          cursor: "pointer",
                          background: "white",
                          fontFamily: '"Edu SA Beginner", cursive',
                        }}
                        {...addHoverHandlers("white", "#f5f5f5")}
                      >
                        Cancel
                      </button>
                    </div>
                  </div>
                )}

                {matchedUser ? (
                  <div
                    style={{
                      marginTop: 14,
                      padding: 14,
                      background: "#eaf8ea",
                      borderRadius: 12,
                      textAlign: "center",
                    }}
                  >
                    <h2 style={{ marginTop: 0, marginBottom: 8, fontSize: 22 }}>
                      Good morning {matchedUser.name} 👋
                    </h2>

                    {manualCorrectionUsed && (
                      <div
                        style={{
                          marginBottom: 10,
                          color: "#555",
                          fontWeight: "bold",
                          fontSize: 14,
                        }}
                      >
                        Manually selected from the list.
                      </div>
                    )}

                    {alreadyCheckedInToday ? (
                      <div
                        style={{
                          marginTop: 8,
                          padding: 10,
                          background: "white",
                          borderRadius: 10,
                          color: "#a00000",
                          fontWeight: "bold",
                          fontSize: 14,
                        }}
                      >
                        {duplicateMessage}
                      </div>
                    ) : !checkInSubmitted ? (
                      <>
                        {!selectionMode || selectionMode === "none" ? (
                          <button
  onClick={() => {
    setMatchedUser(null);
    setManualCorrectionUsed(false);
    setSelectionMode("top6");
    setStatus("Choose your photo");
  }}
  style={{
    marginBottom: 10,
    padding: "8px 12px",
    borderRadius: 10,
    border: "1px solid #ccc",
    cursor: "pointer",
    background: "white",
    fontFamily: '"Edu SA Beginner", cursive',
  }}
  {...addHoverHandlers("white", "#f5f5f5")}
>
  Not me
</button>
                        ) : null}

                        {selectionMode === "none" && (
                          <>
                            <p
                              style={{
                                fontSize: 17,
                                marginBottom: 10,
                                marginTop: 0,
                              }}
                            >
                              How are you feeling today?
                            </p>
                            <div
                              style={{
                                display: "grid",
                                gridTemplateColumns: `repeat(${feelingsColumns}, 1fr)`,
                                gap: 10,
                                marginTop: 8,
                              }}
                            >
                              {FEELINGS.map((feeling) => (
                                <button
                                  key={feeling.label}
                                  onClick={() => submitFeeling(feeling.label)}
                                  style={{
                                    padding: "10px",
                                    borderRadius: 12,
                                    border: "1px solid #ccc",
                                    cursor: "pointer",
                                    background: "white",
                                    display: "flex",
                                    alignItems: "center",
                                    justifyContent: "center",
                                    gap: 8,
                                    fontSize: 14,
                                    fontWeight: 500,
                                    fontFamily: '"Edu SA Beginner", cursive',
                                  }}
                                  {...addHoverHandlers("white", "#f5f5f5")}
                                >
                                  <span style={{ fontSize: 18 }}>
                                    {feeling.emoji}
                                  </span>
                                  <span>{feeling.label}</span>
                                </button>
                              ))}
                            </div>
                          </>
                        )}
                      </>
                    ) : (
                      <div
                        style={{
                          marginTop: 8,
                          padding: 10,
                          background: "white",
                          borderRadius: 10,
                        }}
                      >
                        Check-in saved: <strong>{selectedFeeling}</strong>
                        {(selectedFeeling === "Worried" ||
                          selectedFeeling === "Sad" ||
                          selectedFeeling === "Angry") && (
                          <div style={{ marginTop: 8, color: "#a00000" }}>
                            A teacher will check in on you shortly.
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                ) : (
                  capturedImage &&
                  selectionMode === "none" && (
                    <div
                      style={{
                        marginTop: 14,
                        padding: 12,
                        background: "#ffecec",
                        borderRadius: 12,
                        textAlign: "center",
                      }}
                    >
                      <h2 style={{ marginTop: 0, marginBottom: 0, fontSize: 20 }}>
                        Press Tap to Start to try again
                      </h2>
                    </div>
                  )
                )}
              </>
            )}
          </div>
        </div>

        <div
          style={{
            marginTop: 16,
            background: "white",
            padding: 12,
            borderRadius: 16,
            boxShadow: "0 4px 12px rgba(0,0,0,0.04)",
          }}
        >
          <button
            onClick={() => setTeacherPanelOpen((prev) => !prev)}
            style={{
              width: "100%",
              padding: "12px 14px",
              borderRadius: 12,
              border: "1px solid #ddd",
              background: "#f7f7f7",
              cursor: "pointer",
              fontSize: 17,
              fontWeight: "bold",
              fontFamily: '"Edu SA Beginner", cursive',
            }}
            {...addHoverHandlers("#f7f7f7", "#eeeeee")}
          >
            Teacher Access {teacherPanelOpen ? "▲" : "▼"}
          </button>

          {teacherPanelOpen && (
            <div style={{ marginTop: 14 }}>
              {!teacherUnlocked ? (
                <>
                  <p style={{ textAlign: "center", color: "#666" }}>
                    Teacher area is locked.
                  </p>
                  <div
                    style={{
                      display: "flex",
                      gap: 10,
                      flexWrap: "wrap",
                      alignItems: "center",
                      justifyContent: "center",
                    }}
                  >
                    <input
                      type="password"
                      value={teacherPasscodeInput}
                      onChange={(e) => setTeacherPasscodeInput(e.target.value)}
                      placeholder="Enter passcode"
                      style={{
                        padding: "10px 12px",
                        borderRadius: 10,
                        border: "1px solid #ccc",
                        fontFamily: '"Edu SA Beginner", cursive',
                      }}
                    />
                    <button
                      onClick={unlockTeacherView}
                      style={{
                        padding: "10px 14px",
                        borderRadius: 10,
                        border: "none",
                        background: "black",
                        color: "white",
                        cursor: "pointer",
                        fontFamily: '"Edu SA Beginner", cursive',
                      }}
                    >
                      Unlock
                    </button>
                  </div>

                  {teacherPasscodeError && (
                    <div
                      style={{
                        marginTop: 10,
                        color: "#a00000",
                        textAlign: "center",
                      }}
                    >
                      {teacherPasscodeError}
                    </div>
                  )}
                </>
              ) : (
                <>
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                      gap: 12,
                      flexWrap: "wrap",
                      marginBottom: 12,
                    }}
                  >
                    <h2 style={{ margin: 0, fontSize: 22 }}>Teacher Console</h2>

                    <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
                      <button
                        onClick={() => {
                          setAllCheckIns([]);
                          localStorage.removeItem("classCheckIns");
                          setRecognitionAttempts(0);
                          resetSelectionUi();
                        }}
                        style={{
                          padding: "10px 14px",
                          borderRadius: 10,
                          border: "1px solid #ccc",
                          background: "white",
                          cursor: "pointer",
                          fontFamily: '"Edu SA Beginner", cursive',
                        }}
                        {...addHoverHandlers("white", "#f5f5f5")}
                      >
                        Clear check-ins
                      </button>

                      <button
                        onClick={lockTeacherView}
                        style={{
                          padding: "10px 14px",
                          borderRadius: 10,
                          border: "none",
                          background: "#444",
                          color: "white",
                          cursor: "pointer",
                          fontFamily: '"Edu SA Beginner", cursive',
                        }}
                      >
                        Lock
                      </button>
                    </div>
                  </div>

                  <div
                    style={{
                      marginTop: 12,
                      marginBottom: 16,
                      padding: 12,
                      background: "#f4f7fb",
                      borderRadius: 12,
                    }}
                  >
                    <h3 style={{ marginTop: 0 }}>Today’s Overview</h3>

                    <div
                      style={{
                        display: "grid",
                        gridTemplateColumns: "repeat(4, 1fr)",
                        gap: 10,
                        textAlign: "center",
                        fontSize: 14,
                      }}
                    >
                      <div>
                        <strong>{dashboardStats.total}</strong>
                        <div>Total</div>
                      </div>

                      <div style={{ color: "#a00000" }}>
                        <strong>{dashboardStats.followUp}</strong>
                        <div>Follow-up</div>
                      </div>

                      <div style={{ color: "#005a9c" }}>
                        <strong>{dashboardStats.manual}</strong>
                        <div>Manual</div>
                      </div>

                      <div style={{ color: "#9b6b00" }}>
                        <strong>{dashboardStats.mismatch}</strong>
                        <div>Mismatches</div>
                      </div>
                    </div>

                    <div style={{ marginTop: 10 }}>
                      <strong>Feelings:</strong>
                      <div
                        style={{
                          marginTop: 6,
                          display: "flex",
                          flexWrap: "wrap",
                          gap: 8,
                        }}
                      >
                        {Object.entries(dashboardStats.feelingsCount).length === 0 ? (
                          <div
                            style={{
                              background: "white",
                              padding: "6px 10px",
                              borderRadius: 8,
                              fontSize: 13,
                              color: "#666",
                            }}
                          >
                            No check-ins yet today
                          </div>
                        ) : (
                          Object.entries(dashboardStats.feelingsCount).map(
                            ([feeling, count]) => (
                              <div
                                key={feeling}
                                style={{
                                  background: "white",
                                  padding: "6px 10px",
                                  borderRadius: 8,
                                  fontSize: 13,
                                }}
                              >
                                {feeling}: {count}
                              </div>
                            )
                          )
                        )}
                      </div>
                    </div>
                  </div>

                  {allCheckIns.length === 0 ? (
                    <p style={{ textAlign: "center", color: "#666" }}>
                      No check-ins saved yet.
                    </p>
                  ) : (
                    allCheckIns.map((item) => (
                      <div
                        key={item.id}
                        style={{
                          padding: 14,
                          marginBottom: 12,
                          background: item.needsFollowUp ? "#fff0f0" : "#f7f7f7",
                          borderRadius: 14,
                          border: item.needsFollowUp
                            ? "1px solid #f2b8b8"
                            : "1px solid #e5e5e5",
                        }}
                      >
                        <div
                          style={{
                            display: "grid",
                            gridTemplateColumns:
                              isNarrow ? "1fr" : "120px 1fr auto",
                            gap: 16,
                            alignItems: "center",
                          }}
                        >
                          {item.capturedImage && (
                            <img
                              src={item.capturedImage}
                              alt={`${item.name} check-in`}
                              style={{
                                width: isNarrow ? "100%" : "120px",
                                height: isNarrow ? "auto" : "120px",
                                aspectRatio: "1 / 1",
                                borderRadius: 12,
                                objectFit: "cover",
                              }}
                            />
                          )}

                          <div style={{ width: "100%" }}>
                            <div
                              style={{
                                fontSize: 17,
                                fontWeight: "bold",
                                marginBottom: 6,
                              }}
                            >
                              {item.name} checked in as {item.feeling}
                            </div>

                            <div
                              style={{
                                marginTop: 4,
                                fontSize: 13,
                                color: "#555",
                              }}
                            >
                              Time: {item.time}
                            </div>

                            {item.bestMatchName && (
                              <div
                                style={{
                                  marginTop: 8,
                                  fontSize: 14,
                                  color: "#555",
                                }}
                              >
                                AI top match: <strong>{item.bestMatchName}</strong>
                                {item.bestMatchScore !== null && (
                                  <> ({item.bestMatchScore}%)</>
                                )}
                              </div>
                            )}

                            {item.secondBestMatchName && (
                              <div
                                style={{
                                  marginTop: 4,
                                  fontSize: 14,
                                  color: "#555",
                                }}
                              >
                                Second place:{" "}
                                <strong>{item.secondBestMatchName}</strong>
                                {item.secondBestMatchScore !== null && (
                                  <> ({item.secondBestMatchScore}%)</>
                                )}
                              </div>
                            )}

                            {item.manualCorrectionUsed && (
                              <div
                                style={{
                                  marginTop: 8,
                                  color: "#005a9c",
                                  fontWeight: "bold",
                                  fontSize: 14,
                                }}
                              >
                                Manual correction used. Final choice:{" "}
                                {item.finalChosenName}
                              </div>
                            )}

                            {item.possibleMismatch && (
                              <div
                                style={{
                                  marginTop: 8,
                                  color: "#9b6b00",
                                  fontWeight: "bold",
                                  fontSize: 14,
                                }}
                              >
                                Possible mismatch: top matches were very close.
                              </div>
                            )}

                            {item.needsFollowUp && (
                              <div
                                style={{
                                  marginTop: 8,
                                  color: "#a00000",
                                  fontWeight: "bold",
                                  fontSize: 14,
                                }}
                              >
                                Follow-up needed
                              </div>
                            )}
                          </div>

                          <div style={{ alignSelf: isNarrow ? "stretch" : "start" }}>
                            <button
                              onClick={() => deleteSingleCheckIn(item.id)}
                              style={{
                                padding: "9px 12px",
                                borderRadius: 10,
                                border: "1px solid #d9b3b3",
                                background: "#fff5f5",
                                color: "#a00000",
                                cursor: "pointer",
                                fontFamily: '"Edu SA Beginner", cursive',
                                width: isNarrow ? "100%" : "auto",
                              }}
                              {...addHoverHandlers("#fff5f5", "#ffe8e8")}
                            >
                              Delete
                            </button>
                          </div>
                        </div>
                      </div>
                    ))
                  )}
                </>
              )}
            </div>
          )}
        </div>

        <style>{`
          @keyframes fadeIn {
            from { opacity: 0; transform: translateY(4px); }
            to { opacity: 1; transform: translateY(0); }
          }
        `}</style>
        <style>{`
  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(4px); }
    to { opacity: 1; transform: translateY(0); }
  }

  .logo {
    position: absolute;
    top: 0;
    left: 0;
    z-index: 10;
  }

  .logo img {
    height: 56px;
    object-fit: contain;
  }

  /* Hide on mobile */
  @media (max-width: 768px) {
    .logo {
      display: none;
    }
  }
`}</style>
      </div>
    </div>
  );
}