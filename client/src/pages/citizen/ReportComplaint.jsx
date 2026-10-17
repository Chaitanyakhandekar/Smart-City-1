import React, { useState, useRef } from "react";
import { useNavigate } from "react-router-dom";
import CitizenLayout from "../../components/citizen/CitizenLayout";
import { complaintApi, aiApi } from "../../api/client";
import {
  ArrowLeft,
  ArrowRight,
  Camera,
  UploadCloud,
  RefreshCw,
  AlertCircle,
  CheckCircle2,
  Cpu,
  MapPin,
  Send,
  Loader2,
  Check,
  X,
  Compass,
  Image as ImageIcon,
  Sparkles
} from "lucide-react";
import toast from "react-hot-toast";

export const ReportComplaint = () => {
  const navigate = useNavigate();
  const [currentStep, setCurrentStep] = useState(1); // 1: Image & AI, 2: Title & Desc, 3: Location, 4: Review

  // File & Image State
  const [imageFile, setImageFile] = useState(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState(null);
  const fileInputRef = useRef(null);

  // AI Classification State
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [aiResult, setAiResult] = useState(null);
  const [aiError, setAiError] = useState(null);

  // Form Fields
  const [category, setCategory] = useState("");
  const [subcategory, setSubcategory] = useState("");
  const [priority, setPriority] = useState("MEDIUM");

  // Title & Description (Step 2)
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");

  // Location (Step 3)
  const [locationAddress, setLocationAddress] = useState("");
  const [locating, setLocating] = useState(false);

  // Submission State
  const [submitting, setSubmitting] = useState(false);

  // Image Selection Handler
  const handleFileSelect = (file) => {
    if (!file) return;

    const validTypes = ["image/jpeg", "image/jpg", "image/png", "image/webp"];
    if (!validTypes.includes(file.type) && !file.name.match(/\.(jpeg|jpg|png|webp)$/i)) {
      toast.error("Please upload a valid image (JPEG, PNG, or WEBP).");
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      toast.error("File size exceeds 10MB limit. Please choose a smaller image.");
      return;
    }

    setImageFile(file);
    const reader = new FileReader();
    reader.onload = () => {
      setImagePreviewUrl(reader.result);
    };
    reader.readAsDataURL(file);

    setAiResult(null);
    setAiError(null);
    analyzeImageWithAI(file);
  };

  // Trigger AI / Gemini Classification
  const analyzeImageWithAI = async (fileToAnalyze) => {
    const targetFile = fileToAnalyze || imageFile;
    if (!targetFile) {
      toast.error("Please upload an image first.");
      return;
    }

    setIsAnalyzing(true);
    setAiError(null);

    try {
      const formData = new FormData();
      formData.append("image", targetFile);
      formData.append("description", description || "");

      const res = await aiApi.analyzeImage(formData);
      if (res.data?.data) {
        const result = res.data.data;
        setAiResult(result);
        setCategory(result.category || "Other");
        setSubcategory(result.subcategory || "General Issue");
        setPriority(result.priority || "MEDIUM");

        if (!title && result.subcategory) {
          setTitle(`${result.subcategory} Reported`);
        }

        toast.success(`AI classified: ${result.category} (${result.priority} priority)`);
      } else {
        throw new Error("No classification result returned from server");
      }
    } catch (err) {
      console.warn("AI Classification error:", err.message);
      setAiError("Unable to automatically identify the issue. Please try another image.");
      toast.error("AI could not classify image. Please retry or upload a clearer photo.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleClearImage = () => {
    setImageFile(null);
    setImagePreviewUrl(null);
    setAiResult(null);
    setAiError(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  // GPS Current Location Detection
  const handleGetCurrentLocation = () => {
    if (!navigator.geolocation) {
      toast.error("Geolocation is not supported by your browser.");
      return;
    }

    setLocating(true);
    navigator.geolocation.getCurrentPosition(
      async (position) => {
        const { latitude, longitude } = position.coords;
        try {
          const response = await fetch(
            `https://nominatim.openstreetmap.org/reverse?format=json&lat=${latitude}&lon=${longitude}&zoom=18&addressdetails=1`
          );
          const data = await response.json();
          if (data && data.display_name) {
            setLocationAddress(data.display_name);
            toast.success("Location pinpointed from GPS!");
          } else {
            setLocationAddress(`Lat: ${latitude.toFixed(6)}, Lon: ${longitude.toFixed(6)}`);
            toast.success("Coordinates acquired!");
          }
        } catch (err) {
          setLocationAddress(`Lat: ${latitude.toFixed(6)}, Lon: ${longitude.toFixed(6)}`);
          toast.success("Coordinates acquired!");
        } finally {
          setLocating(false);
        }
      },
      (error) => {
        console.warn("Geolocation error:", error);
        toast.error("Unable to retrieve your location. Please enter manually.");
        setLocating(false);
      },
      { enableHighAccuracy: true, timeout: 10000, maximumAge: 60000 }
    );
  };

  const handleNextFromStep1 = () => {
    if (!imageFile) {
      toast.error("Please upload an image first.");
      return;
    }
    setCurrentStep(2);
  };

  const handleNextFromStep2 = () => {
    if (!title.trim()) {
      toast.error("Please enter a title for the complaint.");
      return;
    }
    if (!description.trim()) {
      toast.error("Please provide a description of the issue.");
      return;
    }
    setCurrentStep(3);
  };

  const handleNextFromStep3 = () => {
    if (!locationAddress.trim()) {
      toast.error("Please provide a location address or landmark.");
      return;
    }
    setCurrentStep(4);
  };

  // Final Submission
  const handleSubmit = async (e) => {
    if (e) e.preventDefault();

    if (!title.trim() || !description.trim() || !locationAddress.trim()) {
      toast.error("Please ensure all complaint fields are filled.");
      return;
    }

    if (!imageFile) {
      toast.error("A photograph of the issue is required.");
      return;
    }

    setSubmitting(true);
    try {
      const formData = new FormData();
      formData.append("title", title.trim());
      formData.append("description", description.trim());
      formData.append("locationAddress", locationAddress.trim());
      formData.append("category", category || aiResult?.category || "Other");
      formData.append("subcategory", subcategory || aiResult?.subcategory || "General Issue");
      formData.append("priority", priority || aiResult?.priority || "MEDIUM");
      formData.append("image", imageFile);

      if (aiResult) {
        formData.append("aiCategory", aiResult.category || category);
        formData.append("aiSubcategory", aiResult.subcategory || subcategory);
        formData.append("aiConfidence", aiResult.confidence || 0.85);
        formData.append("aiPriority", aiResult.priority || priority);
      }

      const res = await complaintApi.createComplaint(formData);
      const createdComplaint = res.data?.data?.complaint;

      toast.success(`Complaint registered: #${createdComplaint?.complaintNumber || "Success"}`);
      navigate(`/citizen/complaints/${createdComplaint?._id}`);
    } catch (err) {
      const message = err.response?.data?.message || "Failed to submit complaint. Please try again.";
      toast.error(message);
    } finally {
      setSubmitting(false);
    }
  };

  const steps = [
    { number: 1, label: "Upload Photo" },
    { number: 2, label: "Details" },
    { number: 3, label: "Location" },
    { number: 4, label: "Review" }
  ];

  return (
    <CitizenLayout
      title="Report a Civic Issue"
      subtitle="AI-powered grievance registration with automated department routing"
      actions={
        <button
          type="button"
          onClick={() => {
            if (currentStep > 1) setCurrentStep(currentStep - 1);
            else navigate("/citizen/dashboard");
          }}
          className="px-3.5 py-1.5 rounded-xl border border-[#E2E8F0] bg-white hover:bg-[#F8FAFC] text-xs font-semibold text-[#0F172A] shadow-2xs flex items-center gap-1.5 transition-colors"
        >
          <ArrowLeft className="w-3.5 h-3.5" /> Back
        </button>
      }
    >
      {/* 4-Step Indicator Bar */}
      <div className="bg-white rounded-2xl p-4 sm:p-5 border border-[#E2E8F0] shadow-xs">
        <div className="flex items-center justify-between max-w-xl mx-auto mb-2 text-xs font-semibold text-[#64748B]">
          <span className="text-[#0D9488] font-bold">Step {currentStep} of {steps.length}</span>
          <span className="text-[11px] bg-[#F8FAFC] text-[#64748B] px-2.5 py-0.5 rounded-full font-medium border border-[#E2E8F0]">
            ~1 min estimated
          </span>
        </div>
        <div className="flex items-center justify-between max-w-xl mx-auto relative pt-1">
          <div className="absolute left-6 right-6 top-5 h-1 bg-[#F1F5F9] -z-0 rounded-full">
            <div
              className="h-full bg-gradient-to-r from-teal-500 to-emerald-500 transition-all duration-300 rounded-full"
              style={{ width: `${((currentStep - 1) / (steps.length - 1)) * 100}%` }}
            />
          </div>

          {steps.map((step) => {
            const isActive = currentStep === step.number;
            const isPassed = currentStep > step.number;
            return (
              <button
                key={step.number}
                type="button"
                onClick={() => {
                  if (step.number < currentStep) setCurrentStep(step.number);
                }}
                className="flex flex-col items-center relative z-10 focus:outline-none group"
              >
                <div
                  className={`w-9 h-9 rounded-2xl flex items-center justify-center font-bold text-xs transition-all ${
                    isActive
                      ? "bg-[#0D9488] text-white ring-4 ring-teal-100 shadow-xs scale-105"
                      : isPassed
                      ? "bg-[#10B981] text-white font-bold"
                      : "bg-[#F8FAFC] text-[#94A3B8] border border-[#E2E8F0]"
                  }`}
                >
                  {isPassed ? <Check className="w-4 h-4 stroke-[3]" /> : step.number}
                </div>
                <span
                  className={`text-[10px] sm:text-xs mt-1.5 font-bold tracking-tight text-center ${
                    isActive ? "text-[#0D9488]" : isPassed ? "text-[#0F172A]" : "text-[#94A3B8]"
                  }`}
                >
                  {step.label}
                </span>
              </button>
            );
          })}
        </div>
      </div>

      {/* ─────────────────────────────────────────────────────────
          STEP 1: UPLOAD PHOTO & AI CLASSIFICATION
         ───────────────────────────────────────────────────────── */}
      {currentStep === 1 && (
        <div className="bg-white rounded-2xl p-5 sm:p-7 border border-[#E2E8F0] shadow-xs space-y-6">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-[#F1F5F9] pb-4">
            <div>
              <h3 className="text-base sm:text-lg font-bold text-[#0F172A]">
                Upload Incident Photograph
              </h3>
              <p className="text-xs text-[#64748B] mt-0.5">
                Our vision model will inspect visual defect signatures to recommend the department and priority.
              </p>
            </div>
            <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-teal-50 text-[#0D9488] text-xs font-semibold w-fit border border-teal-200">
              <Sparkles className="w-3.5 h-3.5 text-[#0D9488]" />
              <span>AI-Assisted</span>
            </div>
          </div>

          {/* Upload Dropzone */}
          {!imagePreviewUrl ? (
            <div
              onClick={() => fileInputRef.current?.click()}
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => {
                e.preventDefault();
                if (e.dataTransfer.files && e.dataTransfer.files[0]) {
                  handleFileSelect(e.dataTransfer.files[0]);
                }
              }}
              className="border-2 border-dashed border-[#CBD5E1] hover:border-[#0D9488] bg-[#F8FAFC] hover:bg-teal-50/20 rounded-2xl p-8 sm:p-12 text-center cursor-pointer transition-all group flex flex-col items-center justify-center space-y-3"
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                capture="environment"
                className="hidden"
                onChange={(e) => {
                  if (e.target.files && e.target.files[0]) {
                    handleFileSelect(e.target.files[0]);
                  }
                }}
              />
              <div className="w-16 h-16 rounded-2xl bg-teal-50 text-[#0D9488] border border-teal-200 flex items-center justify-center group-hover:scale-105 transition-all shadow-xs">
                <Camera className="w-8 h-8" />
              </div>
              <div>
                <p className="text-sm sm:text-base font-bold text-[#0F172A]">
                  Take photo or upload from device
                </p>
                <p className="text-xs text-[#64748B] mt-1 max-w-sm mx-auto">
                  Supports JPG, PNG, WEBP up to 10MB. Clear close-ups ensure the highest classification accuracy.
                </p>
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="relative border border-[#E2E8F0] rounded-2xl overflow-hidden bg-slate-900 aspect-video max-h-80 flex items-center justify-center shadow-xs">
                <img
                  src={imagePreviewUrl}
                  alt="Selected preview"
                  className="w-full h-full object-contain"
                />

                {/* Laser Scanning Animation Beam */}
                {isAnalyzing && (
                  <>
                    <div className="laser-beam animate-scan" />
                    <div className="absolute inset-0 bg-teal-950/20 backdrop-blur-[1px] pointer-events-none" />
                    <div className="absolute top-4 left-4 inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-slate-900/90 border border-teal-400/40 text-teal-300 text-xs font-mono backdrop-blur-md shadow-lg">
                      <span className="w-2 h-2 rounded-full bg-teal-400 animate-ping" />
                      <span>AI SCANNER: ANALYZING PIXELS...</span>
                    </div>
                  </>
                )}

                <button
                  type="button"
                  onClick={handleClearImage}
                  className="absolute top-3 right-3 p-2 rounded-xl bg-slate-900/80 hover:bg-rose-600 text-white shadow-md transition-colors border border-white/20 backdrop-blur-xs z-10"
                  title="Remove image"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>

              {!aiResult && !isAnalyzing && !aiError && (
                <div className="flex justify-center">
                  <button
                    type="button"
                    onClick={() => analyzeImageWithAI(imageFile)}
                    className="px-6 py-2.5 rounded-xl bg-[#0D9488] hover:bg-[#0F766E] text-white font-bold text-xs shadow-xs transition-all flex items-center gap-2"
                  >
                    <Sparkles className="w-4 h-4" /> Analyze Image with AI
                  </button>
                </div>
              )}
            </div>
          )}

          {/* AI Analyzing indicator */}
          {isAnalyzing && (
            <div className="p-4 rounded-xl bg-teal-50 border border-teal-200 flex items-center gap-3 text-[#0D9488]">
              <Loader2 className="w-5 h-5 animate-spin text-[#0D9488] flex-shrink-0" />
              <div>
                <p className="text-xs font-bold text-[#0F172A]">Vision Model Processing...</p>
                <p className="text-[11px] text-[#64748B]">Identifying civic defect signature, category, and severity priority rating...</p>
              </div>
            </div>
          )}

          {/* AI Success Card */}
          {aiResult && !isAnalyzing && (
            <div className="p-5 rounded-2xl bg-teal-50/60 border border-teal-200 space-y-4 shadow-xs">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center gap-2 text-[#0D9488] font-bold text-xs uppercase tracking-wider">
                  <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                  <span>Defect Confirmed by AI Vision</span>
                </div>
                <span className="px-3 py-1 rounded-full text-xs font-bold bg-white text-[#0D9488] border border-teal-200 shadow-2xs">
                  {Math.round((aiResult.confidence || 0.85) * 100)}% Match
                </span>
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 text-xs">
                <div className="bg-white p-3 rounded-xl border border-teal-100 shadow-2xs">
                  <span className="text-[10px] text-[#64748B] font-bold uppercase tracking-wider">Category</span>
                  <p className="font-extrabold text-[#0F172A] mt-1 text-xs truncate">{aiResult.category}</p>
                </div>
                <div className="bg-white p-3 rounded-xl border border-teal-100 shadow-2xs">
                  <span className="text-[10px] text-[#64748B] font-bold uppercase tracking-wider">Subcategory</span>
                  <p className="font-extrabold text-[#0F172A] mt-1 text-xs truncate">{aiResult.subcategory || "General Issue"}</p>
                </div>
                <div className="bg-white p-3 rounded-xl border border-teal-100 shadow-2xs">
                  <span className="text-[10px] text-[#64748B] font-bold uppercase tracking-wider">Priority</span>
                  <p className={`font-extrabold mt-1 text-xs truncate ${
                    aiResult.priority === "HIGH" ? "text-rose-600" : aiResult.priority === "MEDIUM" ? "text-amber-600" : "text-[#0D9488]"
                  }`}>
                    {aiResult.priority || "MEDIUM"}
                  </p>
                </div>
                <div className="bg-white p-3 rounded-xl border border-teal-100 shadow-2xs">
                  <span className="text-[10px] text-[#64748B] font-bold uppercase tracking-wider">Triage</span>
                  <p className="font-extrabold text-emerald-700 mt-1 text-xs truncate">Auto-Assigned</p>
                </div>
              </div>

              <div className="flex items-center justify-between pt-3 border-t border-teal-200/60">
                <button
                  type="button"
                  onClick={handleClearImage}
                  className="px-4 py-2 rounded-xl border border-[#CBD5E1] bg-white text-[#64748B] text-xs font-semibold hover:bg-[#F8FAFC]"
                >
                  Change Photo
                </button>
                <button
                  type="button"
                  onClick={handleNextFromStep1}
                  className="px-6 py-2.5 rounded-xl bg-[#0D9488] hover:bg-[#0F766E] text-white font-bold text-xs shadow-xs flex items-center gap-1.5 transition-colors"
                >
                  Continue to Details <ArrowRight className="w-4 h-4" />
                </button>
              </div>
            </div>
          )}

          {/* AI Failure Card */}
          {aiError && !isAnalyzing && (
            <div className="p-4 rounded-xl bg-amber-50 border border-amber-200 text-[#92400E] space-y-3">
              <div className="flex items-center gap-2 font-bold text-xs text-amber-800">
                <AlertCircle className="w-4 h-4 text-amber-600" />
                {aiError}
              </div>
              <p className="text-xs text-amber-700">
                The photo could not be classified with certainty. Please upload a clearer close-up photograph of the issue.
              </p>
              <div className="flex items-center gap-2 pt-1">
                <button
                  type="button"
                  onClick={() => analyzeImageWithAI(imageFile)}
                  className="px-4 py-2 rounded-xl bg-[#D97706] hover:bg-[#B45309] text-white text-xs font-bold shadow-2xs flex items-center gap-1.5"
                >
                  <RefreshCw className="w-3.5 h-3.5" /> Retry Analysis
                </button>
                <button
                  type="button"
                  onClick={handleClearImage}
                  className="px-4 py-2 rounded-xl border border-amber-300 bg-white text-amber-800 text-xs font-semibold hover:bg-amber-50"
                >
                  Replace Image
                </button>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ─────────────────────────────────────────────────────────
          STEP 2: TITLE + DESCRIPTION
         ───────────────────────────────────────────────────────── */}
      {currentStep === 2 && (
        <div className="bg-white rounded-2xl p-6 sm:p-7 border border-[#E2E8F0] shadow-xs space-y-6">
          <div>
            <h3 className="text-base sm:text-lg font-bold text-[#0F172A]">
              Tell us about the issue
            </h3>
            <p className="text-xs text-[#64748B] mt-0.5">
              Give your complaint a concise title and details to guide field crews.
            </p>
          </div>

          {/* AI Detection Summary */}
          <div className="p-3.5 rounded-xl bg-[#F8FAFC] border border-[#E2E8F0] flex items-center justify-between text-xs">
            <div className="flex items-center gap-2">
              <Cpu className="w-4 h-4 text-[#0D9488] flex-shrink-0" />
              <span className="text-[#64748B] font-semibold">AI Detected:</span>
              <span className="font-bold text-[#0F172A]">
                {category} → {subcategory}
              </span>
            </div>
            <span className="px-2.5 py-0.5 rounded-full bg-teal-50 text-[#0D9488] text-[10px] font-bold border border-teal-200">
              {priority} Priority
            </span>
          </div>

          <div>
            <label className="block text-xs font-bold uppercase tracking-wider text-[#0F172A] mb-1">
              Complaint Title *
            </label>
            <input
              type="text"
              required
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="e.g. Pothole near school gate"
              className="w-full px-4 py-2.5 bg-[#F8FAFC] border border-[#E2E8F0] rounded-xl text-sm text-[#0F172A] placeholder-[#94A3B8] focus:outline-none focus:border-[#0D9488] focus:bg-white"
            />
          </div>

          <div>
            <div className="flex items-center justify-between mb-1">
              <label className="text-xs font-bold uppercase tracking-wider text-[#0F172A]">
                Description *
              </label>
              <span className="text-[11px] text-[#94A3B8]">
                {description.length}/500
              </span>
            </div>
            <textarea
              required
              rows={4}
              maxLength={500}
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="e.g. Large pothole near the school entrance. It is dangerous for vehicles and pedestrians."
              className="w-full px-4 py-2.5 bg-[#F8FAFC] border border-[#E2E8F0] rounded-xl text-sm text-[#0F172A] placeholder-[#94A3B8] focus:outline-none focus:border-[#0D9488] focus:bg-white resize-y"
            />
          </div>

          <div className="flex items-center justify-between pt-3 border-t border-[#F1F5F9]">
            <button
              type="button"
              onClick={() => setCurrentStep(1)}
              className="px-5 py-2.5 rounded-xl border border-[#E2E8F0] bg-white text-[#64748B] text-xs font-semibold hover:bg-[#F8FAFC]"
            >
              Back
            </button>
            <button
              type="button"
              onClick={handleNextFromStep2}
              className="px-7 py-2.5 rounded-xl bg-[#0D9488] hover:bg-[#0F766E] text-white font-bold text-xs shadow-xs flex items-center gap-1.5 transition-colors"
            >
              Continue <ArrowRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      )}

      {/* ─────────────────────────────────────────────────────────
          STEP 3: LOCATION
         ───────────────────────────────────────────────────────── */}
      {currentStep === 3 && (
        <div className="bg-white rounded-2xl p-6 sm:p-7 border border-[#E2E8F0] shadow-xs space-y-6">
          <div>
            <h3 className="text-base sm:text-lg font-bold text-[#0F172A]">
              Location & Area Landmark *
            </h3>
            <p className="text-xs text-[#64748B] mt-0.5">
              Provide exact streets, area name, or nearby landmark for field inspectors.
            </p>
          </div>

          <div>
            <label className="block text-xs font-bold uppercase tracking-wider text-[#0F172A] mb-1 flex items-center gap-1.5">
              <MapPin className="w-4 h-4 text-[#0D9488]" />
              Address / Nearby Landmark *
            </label>
            <input
              type="text"
              required
              value={locationAddress}
              onChange={(e) => setLocationAddress(e.target.value)}
              placeholder="e.g. North Avenue, Sector 4 (Near Metro Station)"
              className="w-full px-4 py-2.5 bg-[#F8FAFC] border border-[#E2E8F0] rounded-xl text-sm text-[#0F172A] placeholder-[#94A3B8] focus:outline-none focus:border-[#0D9488] focus:bg-white"
            />
          </div>

          {/* Current Location Button */}
          <div className="flex flex-wrap items-center gap-3">
            <button
              type="button"
              onClick={handleGetCurrentLocation}
              disabled={locating}
              className="px-4 py-2 rounded-xl border border-teal-200 bg-teal-50 text-[#0D9488] hover:bg-teal-100/60 text-xs font-bold flex items-center gap-2 transition-colors"
            >
              {locating ? <Loader2 className="w-4 h-4 animate-spin" /> : <Compass className="w-4 h-4 text-[#0D9488]" />}
              {locating ? "Acquiring GPS Location..." : "Use Device GPS Location"}
            </button>
            <span className="text-[11px] text-[#64748B]">
              Auto-detects GPS coordinates for priority routing
            </span>
          </div>

          <div className="p-4 rounded-xl bg-[#F8FAFC] border border-[#E2E8F0] flex items-start gap-3">
            <MapPin className="w-5 h-5 text-[#0D9488] flex-shrink-0 mt-0.5" />
            <div className="text-xs text-[#475569]">
              <span className="font-bold text-[#0F172A]">Dispatch Notice:</span> Municipal response crews prioritize complaints with verified area landmarks and GPS coordinates.
            </div>
          </div>

          <div className="flex items-center justify-between pt-3 border-t border-[#F1F5F9]">
            <button
              type="button"
              onClick={() => setCurrentStep(2)}
              className="px-5 py-2.5 rounded-xl border border-[#E2E8F0] bg-white text-[#64748B] text-xs font-semibold hover:bg-[#F8FAFC]"
            >
              Back
            </button>
            <button
              type="button"
              onClick={handleNextFromStep3}
              className="px-7 py-2.5 rounded-xl bg-[#0D9488] hover:bg-[#0F766E] text-white font-bold text-xs shadow-xs flex items-center gap-1.5 transition-colors"
            >
              Continue to Review <ArrowRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      )}

      {/* ─────────────────────────────────────────────────────────
          STEP 4: REVIEW & SUBMIT
         ───────────────────────────────────────────────────────── */}
      {currentStep === 4 && (
        <div className="bg-white rounded-2xl p-6 sm:p-7 border border-[#E2E8F0] shadow-xs space-y-6">
          <div>
            <h3 className="text-base sm:text-lg font-bold text-[#0F172A]">
              Review Complaint
            </h3>
            <p className="text-xs text-[#64748B] mt-0.5">
              Confirm the details below before submitting to municipal operations.
            </p>
          </div>

          <div className="border border-[#E2E8F0] rounded-xl p-5 bg-[#F8FAFC] space-y-4">
            {/* Image Preview */}
            {imagePreviewUrl && (
              <div>
                <span className="text-[10px] font-bold uppercase tracking-wider text-[#64748B]">
                  Uploaded Photo
                </span>
                <div className="mt-1.5 w-36 h-28 rounded-xl overflow-hidden border border-[#E2E8F0] shadow-2xs">
                  <img
                    src={imagePreviewUrl}
                    alt="Complaint proof"
                    className="w-full h-full object-cover"
                  />
                </div>
              </div>
            )}

            {/* AI Classification Block */}
            <div className="p-4 rounded-xl bg-white border border-[#E2E8F0] space-y-2 shadow-2xs">
              <span className="text-[10px] font-bold uppercase tracking-wider text-[#0D9488] flex items-center gap-1.5">
                <Cpu className="w-3.5 h-3.5" /> AI Classification
              </span>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs pt-1">
                <div>
                  <p className="text-[10px] text-[#64748B] font-semibold uppercase">Category</p>
                  <p className="font-bold text-[#0F172A]">{category}</p>
                </div>
                <div>
                  <p className="text-[10px] text-[#64748B] font-semibold uppercase">Subcategory</p>
                  <p className="font-bold text-[#0F172A]">{subcategory || "General"}</p>
                </div>
                <div>
                  <p className="text-[10px] text-[#64748B] font-semibold uppercase">Confidence</p>
                  <p className="font-bold text-[#0D9488]">{Math.round((aiResult?.confidence || 0.85) * 100)}%</p>
                </div>
                <div>
                  <p className="text-[10px] text-[#64748B] font-semibold uppercase">Priority</p>
                  <p className="font-bold text-amber-700">{priority}</p>
                </div>
              </div>
            </div>

            <div>
              <span className="text-[10px] font-bold uppercase tracking-wider text-[#64748B]">
                Title
              </span>
              <p className="text-sm font-bold text-[#0F172A] mt-0.5">{title}</p>
            </div>

            <div>
              <span className="text-[10px] font-bold uppercase tracking-wider text-[#64748B]">
                Description
              </span>
              <p className="text-xs text-[#475569] mt-0.5 leading-relaxed">{description}</p>
            </div>

            <div>
              <span className="text-[10px] font-bold uppercase tracking-wider text-[#64748B]">
                Location
              </span>
              <p className="text-xs font-semibold text-[#0D9488] flex items-center gap-1 mt-0.5">
                <MapPin className="w-3.5 h-3.5 text-[#0D9488]" />
                {locationAddress}
              </p>
            </div>
          </div>

          <div className="flex items-center justify-between pt-3 border-t border-[#F1F5F9]">
            <button
              type="button"
              onClick={() => setCurrentStep(3)}
              className="px-5 py-2.5 rounded-xl border border-[#E2E8F0] bg-white text-[#64748B] text-xs font-semibold hover:bg-[#F8FAFC]"
            >
              Back
            </button>

            <button
              type="button"
              onClick={handleSubmit}
              disabled={submitting}
              className="px-7 py-2.5 rounded-xl bg-[#0D9488] hover:bg-[#0F766E] text-white font-bold text-xs shadow-xs disabled:opacity-50 transition-colors flex items-center gap-2"
            >
              {submitting ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" /> Submitting...
                </>
              ) : (
                <>
                  <Send className="w-4 h-4" /> Submit Complaint
                </>
              )}
            </button>
          </div>
        </div>
      )}
    </CitizenLayout>
  );
};

export default ReportComplaint;
