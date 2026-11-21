import React, { useState, useEffect } from 'react';
import { Navbar, NavTab } from './components/Navbar';
import { Footer } from './components/Footer';
import { MedicalDisclaimer } from './components/MedicalDisclaimer';
import { LandingPage } from './pages/LandingPage';
import { PredictorPage } from './pages/PredictorPage';
import { LabReportPage } from './pages/LabReportPage';
import { PerformancePage } from './pages/PerformancePage';
import { ComparisonPage } from './pages/ComparisonPage';
import { AboutPage } from './pages/AboutPage';
import { apiService } from './services/api';
import { LabReportRecord } from './types';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<NavTab>('landing');
  const [systemHealthy, setSystemHealthy] = useState<boolean>(true);
  const [latestReport, setLatestReport] = useState<LabReportRecord | null>(null);
  const [reportHistory, setReportHistory] = useState<LabReportRecord[]>([]);

  useEffect(() => {
    const verifyHealth = async () => {
      try {
        const res = await apiService.checkHealth();
        setSystemHealthy(res.status === 'healthy');
      } catch (err) {
        console.warn('Backend API currently unreachable:', err);
        setSystemHealthy(false);
      }
    };
    verifyHealth();
    const interval = setInterval(verifyHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  const handleSaveReport = (record: LabReportRecord) => {
    setLatestReport(record);
    setReportHistory((prev) => [record, ...prev]);
  };

  const renderActiveTab = () => {
    switch (activeTab) {
      case 'landing':
        return <LandingPage setActiveTab={(tab) => setActiveTab(tab)} />;
      case 'predictor':
        return (
          <PredictorPage
            onSaveReport={handleSaveReport}
            onOpenReport={() => setActiveTab('report')}
          />
        );
      case 'report':
        return (
          <LabReportPage
            latestReport={latestReport}
            reportHistory={reportHistory}
            onSelectReport={(rec) => setLatestReport(rec)}
            setActiveTab={(tab) => setActiveTab(tab)}
          />
        );
      case 'performance':
        return <PerformancePage />;
      case 'comparison':
        return <ComparisonPage />;
      case 'about':
        return <AboutPage />;
      default:
        return <LandingPage setActiveTab={(tab) => setActiveTab(tab)} />;
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-sky-500/30 selection:text-sky-200 print:bg-white print:text-slate-900">
      <Navbar 
        activeTab={activeTab} 
        setActiveTab={setActiveTab} 
        systemHealthy={systemHealthy}
        hasReport={latestReport !== null}
      />
      
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6 print:p-0 print:max-w-none">
        <MedicalDisclaimer />
        {renderActiveTab()}
      </main>

      <Footer />
    </div>
  );
};

export default App;
