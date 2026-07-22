import pandas as pd


class WorkbookExporter:

    @staticmethod
    def write_stage_sheet(
        writer,
        stage_name,
        final_df
    ):
        final_df.to_excel(
            writer,
            sheet_name=stage_name[:31],
            index=False
        )

    @staticmethod
    def write_xml_sheet(
        writer,
        stage_xml_exports
    ):

        if stage_xml_exports:

            pd.DataFrame(
                stage_xml_exports
            ).to_excel(
                writer,
                sheet_name='CurveData_XML',
                index=False
            )

    @staticmethod
    def write_operating_conditions(
        writer,
        property_rows
    ):

        if property_rows:

            pd.DataFrame(
                property_rows,
                columns=[
                    'Stage',
                    'Parameter',
                    'Value',
                    'Units'
                ]
            ).to_excel(
                writer,
                sheet_name='Operating_Conditions',
                index=False
            )

    @staticmethod
    def write_overview(writer,overview):

        pd.DataFrame(overview).to_excel(writer,sheet_name='Workbook_Overview',index=False)

    @staticmethod
    def write_r2_summary(writer,r2_rows):

        pd.DataFrame(r2_rows,columns=['Stage','Speed','Parameter','Method','R2']).to_excel(writer,sheet_name='Summary_R2',index=False)

    @staticmethod
    def write_scaling_factors(
        writer,
        scaling_rows
    ):

        if scaling_rows:

            pd.DataFrame(
                scaling_rows
            ).to_excel(
                writer,
                sheet_name='Scaling_Factors',
                index=False
            )

    @staticmethod
    def write_side_stream_sheet(
        writer,
        side_stream_df
    ):

        if not side_stream_df.empty:

            side_stream_df.to_excel(
                writer,
                sheet_name='SideStreamPerformanceData',
                index=False
            )

    @classmethod
    def write_all_summary_sheets(
        cls,
        writer,
        stage_xml_exports,
        property_rows,
        overview,
        r2_rows,
        scaling_rows=None
    ):

        cls.write_xml_sheet(
            writer,
            stage_xml_exports
        )

        cls.write_operating_conditions(
            writer,
            property_rows
        )

        cls.write_overview(
            writer,
            overview
        )

        cls.write_r2_summary(
            writer,
            r2_rows
        )

        if scaling_rows:
            cls.write_scaling_factors(
                writer,
                scaling_rows
            )