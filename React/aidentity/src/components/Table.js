import React from "react";
import { styled } from "@mui/material/styles";
import Table from "@mui/material/Table";
import TableBody from "@mui/material/TableBody";
import TableCell, { tableCellClasses } from "@mui/material/TableCell";
import TableContainer from "@mui/material/TableContainer";
import TableHead from "@mui/material/TableHead";
import TableRow from "@mui/material/TableRow";
import Paper from "@mui/material/Paper";
import Button from "@mui/material/Button";

const StyledTableCell = styled(TableCell)(({ theme }) => ({
  [`&.${tableCellClasses.head}`]: {
    backgroundColor: "#1a353e",
    color: theme.palette.common.white,
  },
  [`&.${tableCellClasses.body}`]: {
    fontSize: 14,
  },
}));

const StyledTableRow = styled(TableRow)(({ theme }) => ({
  "&:nth-of-type(odd)": {
    backgroundColor: theme.palette.action.hover,
  },
  // hide last border
  "&:last-child td, &:last-child th": {
    border: 0,
  },
}));

export default function CustomizedTable({
  models,
  headerText,
  onSetActiveModel,
  isAllModelsTable,
  activeModel,
}) {
  return (
    <TableContainer component={Paper} sx={{ marginBottom: "2rem" }}>
      <Table aria-label="customized table">
        <TableHead>
          <TableRow>
            <StyledTableCell>{headerText}</StyledTableCell>
            {isAllModelsTable && <StyledTableCell>Actions</StyledTableCell>}
          </TableRow>
        </TableHead>
        <TableBody>
          {models.map((model, index) => (
            <StyledTableRow key={index}>
              <StyledTableCell component="th" scope="row">
                {model}
              </StyledTableCell>
              {isAllModelsTable && (
                <StyledTableCell>
                  <Button
                    variant="outlined"
                    onClick={() => onSetActiveModel(model)}
                    disabled={model === activeModel}
                  >
                    Set Active
                  </Button>
                </StyledTableCell>
              )}
            </StyledTableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );
}
