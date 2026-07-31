from models import RuleEngineData


class HSEPromptGenerator:
    """
    Generates LLM prompts for HSE (Health, Safety, and Environment) safety reports.

    Prompts are strictly based on Rule Engine output. Each method returns a
    ready-to-send prompt string that instructs the LLM to act as a specific
    HSE persona, use only the supplied data (no inference or speculation),
    and return its response in a fixed Markdown section structure so
    downstream report rendering (e.g. Jinja2 templates) can rely on a
    consistent format.
    """

    @classmethod
    def daily_report(cls, data: RuleEngineData) -> str:
        """
        Build a prompt for a daily safety report.

        Args:
            data: Rule Engine output containing the day's detections,
                violations, and statistics to summarize.

        Returns:
            A prompt string instructing the LLM to produce a Markdown
            report with Summary, Statistics, Key Violations, Risk
            Assessment, and Recommendations sections, based only on
            the provided data.
        """
        return f"""
        You are an industrial HSE expert.

        Generate a professional daily safety report.

        Use only the provided Rule Engine data.
        Do not infer or invent information.

        Return Markdown format:

        # Daily Safety Report

        ## Summary
        ## Statistics
        ## Key Violations
        ## Risk Assessment
        ## Recommendations

        Data:
        {data}
        """

    @classmethod
    def weekly_report(cls, data: RuleEngineData) -> str:
        """
        Build a prompt for a weekly safety report.

        Args:
            data: Rule Engine output aggregated over the week, used to
                identify trends and repeated violations.

        Returns:
            A prompt string instructing the LLM to produce a Markdown
            report with Summary, Trends, Frequent Violations, Risk
            Areas, and Recommendations sections, based only on the
            provided data.
        """
        return f"""
        You are an industrial HSE analyst.

        Generate a weekly safety report.

        Analyze:
        - safety trends
        - repeated violations
        - overall safety performance

        Use only the provided data.
        Do not invent information.

        Return Markdown format:

        # Weekly Safety Report

        ## Summary
        ## Trends
        ## Frequent Violations
        ## Risk Areas
        ## Recommendations

        Data:
        {data}
        """

    @classmethod
    def incident_report(cls, data: RuleEngineData) -> str:
        """
        Build a prompt for a single-incident report.

        Args:
            data: Rule Engine output describing the specific incident
                to be summarized.

        Returns:
            A prompt string instructing the LLM to produce a factual
            Markdown report with Summary, Facts, Risks, Immediate
            Actions, and Recommendations sections, explicitly avoiding
            speculation, root-cause assumptions, or blame assignment.
        """
        return f"""
        You are an HSE incident investigator.

        Summarize the incident using only the provided data.

        Do not speculate, assume causes, or assign blame.

        Return Markdown format:

        # Incident Report

        ## Summary
        ## Facts
        ## Risks
        ## Immediate Actions
        ## Recommendations

        Data:
        {data}
        """

    @classmethod
    def executive_summary(cls, data: RuleEngineData) -> str:
        """
        Build a prompt for a concise executive summary.

        Args:
            data: Rule Engine output to be condensed into a short,
                high-level summary for leadership review.

        Returns:
            A prompt string instructing the LLM to produce a
            200-word-max Markdown summary with Overall Status,
            Critical Findings, and up to three Recommendations,
            based only on the provided data.
        """
        return f"""
        You are an HSE advisor.

        Write a concise executive summary.
        Maximum length: 200 words.

        Highlight:
        - most important findings
        - up to three recommendations

        Use only the provided data.

        Return Markdown format:

        # Executive Summary

        ## Overall Status
        ## Critical Findings
        ## Recommendations

        Data:
        {data}
        """