import React, { useCallback, useEffect, useState } from 'react';
import { FlaskConical, Play, RefreshCw } from 'lucide-react';
import { Button } from '../components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { ScrollArea } from '../components/ui/scroll-area';
import { toast } from 'sonner';
import {
  listMemoryScenarios,
  runMemoryScenarios,
  getMemoryScenarioReport,
  getMemoryScenariosLatest,
} from '../lib/api';

export default function MemoryLab() {
  const [scenarios, setScenarios] = useState([]);
  const [dashboard, setDashboard] = useState(null);
  const [selectedId, setSelectedId] = useState(null);
  const [report, setReport] = useState(null);
  const [promptText, setPromptText] = useState('');
  const [loading, setLoading] = useState(false);
  const [running, setRunning] = useState(false);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const data = await listMemoryScenarios();
      setScenarios(data.scenarios || []);
      setDashboard(data.last_dashboard || (await getMemoryScenariosLatest()));
    } catch (e) {
      toast.error(String(e.message || e));
    } finally {
      setLoading(false);
    }
  }, []);

  const loadReport = useCallback(async (id) => {
    setSelectedId(id);
    try {
      const data = await getMemoryScenarioReport(id);
      setReport(data);
      setPromptText(data.report_md || '');
    } catch (e) {
      toast.error(String(e.message || e));
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const runAll = async () => {
    setRunning(true);
    try {
      const result = await runMemoryScenarios({ all: true });
      toast.success(result.ok ? 'All scenarios passed' : 'Some scenarios failed');
      await refresh();
      if (result.scenario_id) loadReport(result.scenario_id);
    } catch (e) {
      toast.error(String(e.message || e));
    } finally {
      setRunning(false);
    }
  };

  const runOne = async (id) => {
    setRunning(true);
    try {
      const result = await runMemoryScenarios({ scenario_id: id });
      toast.success(result.ok ? `${id} passed` : `${id} failed`);
      await refresh();
      loadReport(id);
    } catch (e) {
      toast.error(String(e.message || e));
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="flex flex-col h-full gap-4 p-4" data-testid="memory-lab-page">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <FlaskConical className="h-5 w-5" />
          <h1 className="text-lg font-semibold">Memory Lab</h1>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={refresh} disabled={loading}>
            <RefreshCw className="h-4 w-4 mr-1" />
            Refresh
          </Button>
          <Button size="sm" onClick={runAll} disabled={running}>
            <Play className="h-4 w-4 mr-1" />
            Run all
          </Button>
        </div>
      </div>

      {dashboard && (
        <Card>
          <CardHeader className="py-3">
            <CardTitle className="text-sm">Last run</CardTitle>
          </CardHeader>
          <CardContent className="text-sm text-muted-foreground py-2">
            {dashboard.passed ?? 0} passed, {dashboard.failed ?? 0} failed
            {dashboard.reports_dir ? ` — ${dashboard.reports_dir}` : ''}
          </CardContent>
        </Card>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 flex-1 min-h-0">
        <Card className="min-h-[320px]">
          <CardHeader className="py-3">
            <CardTitle className="text-sm">Scenarios</CardTitle>
          </CardHeader>
          <CardContent className="p-0">
            <ScrollArea className="h-[360px]">
              <ul className="divide-y">
                {scenarios.map((s) => (
                  <li
                    key={s.id}
                    className={`px-4 py-2 flex justify-between items-start gap-2 cursor-pointer hover:bg-muted/50 ${
                      selectedId === s.id ? 'bg-muted' : ''
                    }`}
                    onClick={() => loadReport(s.id)}
                  >
                    <div>
                      <div className="font-medium text-sm">{s.file || s.id}</div>
                      <div className="text-xs text-muted-foreground line-clamp-2">{s.description}</div>
                    </div>
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={(e) => {
                        e.stopPropagation();
                        runOne(s.id);
                      }}
                      disabled={running}
                    >
                      Run
                    </Button>
                  </li>
                ))}
              </ul>
            </ScrollArea>
          </CardContent>
        </Card>

        <Card className="min-h-[320px]">
          <CardHeader className="py-3">
            <CardTitle className="text-sm">
              Report {selectedId ? `— ${selectedId}` : ''}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ScrollArea className="h-[360px]">
              <pre className="text-xs whitespace-pre-wrap font-mono">
                {report?.report_md || promptText || 'Select a scenario to view report.md'}
              </pre>
              {report?.result?.failures?.length > 0 && (
                <div className="mt-4 text-sm text-destructive">
                  {report.result.failures.map((f) => (
                    <div key={f}>{f}</div>
                  ))}
                </div>
              )}
            </ScrollArea>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
