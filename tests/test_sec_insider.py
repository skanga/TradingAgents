import pytest

from tradingagents.dataflows import sec_insider


# --- Fixture: a minimal but realistic Form 4 XML document ---

_FORM4_XML = """<?xml version="1.0"?>
<ownershipDocument>
  <reportingOwner>
    <reportingOwnerId>
      <rptOwnerName>Cook, Timothy D.</rptOwnerName>
    </reportingOwnerId>
    <reportingOwnerRelationship>
      <isOfficer>1</isOfficer>
      <officerTitle>Chief Executive Officer</officerTitle>
      <isDirector>1</isDirector>
      <isTenPercentOwner>0</isTenPercentOwner>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <securityTitle><value>Common Stock</value></securityTitle>
      <transactionDate><value>2026-04-15</value></transactionDate>
      <transactionCoding>
        <transactionCode>P</transactionCode>
      </transactionCoding>
      <transactionAmounts>
        <transactionShares><value>4000</value></transactionShares>
        <transactionPricePerShare><value>187.50</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
      <postTransactionAmounts>
        <sharesOwnedFollowingTransaction><value>54000</value></sharesOwnedFollowingTransaction>
      </postTransactionAmounts>
    </nonDerivativeTransaction>
    <nonDerivativeTransaction>
      <securityTitle><value>Common Stock</value></securityTitle>
      <transactionDate><value>2026-04-12</value></transactionDate>
      <transactionCoding>
        <transactionCode>S</transactionCode>
      </transactionCoding>
      <transactionAmounts>
        <transactionShares><value>1000</value></transactionShares>
        <transactionPricePerShare><value>185.20</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
      <postTransactionAmounts>
        <sharesOwnedFollowingTransaction><value>49000</value></sharesOwnedFollowingTransaction>
      </postTransactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>
"""


def test_parse_form4_xml_extracts_purchase_and_sale():
    txns = sec_insider._parse_form4_xml(_FORM4_XML)
    assert len(txns) == 2

    buy, sell = txns
    assert buy["filer"] == "Cook, Timothy D."
    assert "Chief Executive Officer" in buy["title"]
    assert "Director" in buy["title"]
    assert buy["code"] == "P"
    assert buy["direction"] == "A"
    assert buy["shares"] == 4000
    assert buy["price"] == 187.50
    assert buy["value"] == 4000 * 187.50

    assert sell["code"] == "S"
    assert sell["direction"] == "D"


def test_parse_form4_xml_handles_malformed_input():
    assert sec_insider._parse_form4_xml("not xml at all") == []


def test_parse_form4_xml_skips_transactions_missing_shares():
    xml = """<?xml version="1.0"?>
<ownershipDocument>
  <reportingOwner>
    <reportingOwnerId><rptOwnerName>Doe, Jane</rptOwnerName></reportingOwnerId>
    <reportingOwnerRelationship><isOfficer>1</isOfficer></reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionPricePerShare><value>10.0</value></transactionPricePerShare>
      </transactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>"""
    assert sec_insider._parse_form4_xml(xml) == []


def test_format_report_emits_cluster_summary_and_large_purchase_flag():
    txns = sec_insider._parse_form4_xml(_FORM4_XML)
    report = sec_insider._format_report("AAPL", txns, lookback_days=90, filing_count=2)

    assert "AAPL" in report
    assert "Cluster summary" in report
    assert "1 unique buyer" in report
    assert "1 unique seller" in report
    assert "net +0 buyers" in report
    # 4000 * 187.50 = $750,000 — above the $500k threshold
    assert "Notable" in report
    assert "Cook, Timothy D." in report
    assert "$750,000" in report
    assert report.startswith("##")  # not a fallback string


def test_format_report_handles_empty_filings():
    report = sec_insider._format_report("XYZ", [], lookback_days=90, filing_count=0)
    assert "No Form 4 filings" in report
    assert not report.startswith("[")


def test_get_insider_transactions_returns_fallback_for_unknown_ticker(monkeypatch):
    # Force the CIK lookup to return None without hitting the network.
    monkeypatch.setattr(sec_insider, "_ticker_to_cik", lambda _t: None)
    out = sec_insider.get_insider_transactions("ZZZZZ", lookback_days=90)
    assert out.startswith("[SEC Form 4: no CIK match")


@pytest.mark.integration
def test_get_insider_transactions_live_aapl():
    """Live SEC EDGAR fetch — opt-in via `pytest -m integration`."""
    out = sec_insider.get_insider_transactions("AAPL", lookback_days=90)
    assert isinstance(out, str) and out
    assert not out.startswith("["), out
    assert "AAPL" in out
