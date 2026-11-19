import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import App from "./App";
import "./index.css";
import { RootErrorBoundary } from "./components/ErrorBoundary";
import { TelemetryProvider } from "./hooks/useTelemetry";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <RootErrorBoundary>
      <BrowserRouter>
        <TelemetryProvider>
          <App />
        </TelemetryProvider>
      </BrowserRouter>
    </RootErrorBoundary>
  </React.StrictMode>,
);
